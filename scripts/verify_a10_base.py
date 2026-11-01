"""校验 A10 Base 开发门禁，不读取 test。

Verify the A10 Base development gate without reading the test split.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.artifacts import RunArtifacts
from llm_lifecycle_lab.config import config_sha256, load_run_config
from llm_lifecycle_lab.data.packing import load_packed_pretraining_manifest
from llm_lifecycle_lab.exceptions import ContractError, LLMLabError
from llm_lifecycle_lab.training.engine import EngineConfig

PIPELINES = {
    "smoke": "native-60m-a10-base-smoke.yaml",
    "pilot": "native-60m-a10-base-pilot100.yaml",
    "base": "native-60m-a10-base-v1.yaml",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def verify(run: Path, *, phase: str, root: Path) -> dict[str, Any]:
    config = load_run_config(root / "configs/pipelines" / PIPELINES[phase])
    digest = config_sha256(config)
    manifest = read_json(run / "run_manifest.json")
    result = read_json(run / "training_result.json")
    runtime = read_json(run / "runtime_environment.json")
    require(manifest["status"] == "completed", "run is not completed")
    require(manifest["run_id"] == run.name, "run directory identity mismatch")
    require(manifest["config_sha256"] == digest, "A10 config identity mismatch")
    require(
        runtime["accelerator"]["type"] == "cuda"
        and len(runtime["accelerator"]["devices"]) == 1
        and runtime["accelerator"]["devices"][0]["name"] == "NVIDIA A10",
        "run must have executed on a single NVIDIA A10",
    )
    if phase == "base":
        require(runtime["code"]["dirty"] is False, "formal Base source was not clean")
        require(bool(runtime["code"]["commit"]), "formal Base has no source commit")

    packed = load_packed_pretraining_manifest(root / config.data["packed_manifest"])
    train = next(split for split in packed.splits if split.name == "train")
    _, budget = EngineConfig.from_dict(config.training).resolve_budget(
        examples_per_epoch=train.examples,
        supervised_tokens_per_epoch=train.supervised_tokens,
    )
    require(result["global_step"] == budget.max_steps, "optimizer budget incomplete")
    require(
        result["target_train_tokens"] == budget.target_train_tokens,
        "supervised-token budget mismatch",
    )
    coverage = float(result["tokens_seen"]) / budget.target_train_tokens
    require(1.0 <= coverage <= 1.01, "token coverage is outside [1.0, 1.01]")
    require(
        math.isclose(coverage, float(result["target_token_coverage"]), abs_tol=1e-12),
        "reported token coverage is inconsistent",
    )

    rows = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = next(row for row in rows if row.get("event") == "baseline")
    evaluated = [row for row in rows if "train_loss" in row and "eval_loss" in row]
    require(bool(evaluated), "run has no training-time dev evaluation")
    final = evaluated[-1]
    require(final["step"] == budget.max_steps, "final step was not evaluated")
    keys = ("eval_loss", "eval_en_loss", "eval_zh_loss")
    for row in (baseline, *evaluated):
        for key in keys:
            require(math.isfinite(float(row[key])), f"non-finite {key}")
    for row in rows:
        if "train_loss" in row:
            for key in ("train_loss", "gradient_norm"):
                require(math.isfinite(float(row[key])), f"non-finite {key}")
    require(
        float(final["eval_loss"]) < float(baseline["eval_loss"]),
        "overall dev loss did not improve",
    )
    if phase != "smoke":
        for key in keys[1:]:
            require(
                float(final[key]) < float(baseline[key]),
                f"{key} did not improve",
            )
        best = min(float(row["eval_loss"]) for row in evaluated)
        require(float(final["eval_loss"]) <= 1.02 * best, "final loss regressed >2%")
    if phase == "base":
        require(
            float(final["eval_loss"]) <= 0.5 * float(baseline["eval_loss"]),
            "formal Base loss did not halve",
        )

    checkpoint = run / "checkpoints" / f"step-{budget.max_steps:08d}"
    metadata = read_json(checkpoint / "checkpoint_metadata.json")
    require(metadata["config_sha256"] == digest, "checkpoint config mismatch")
    require(metadata["run_id"] == run.name, "checkpoint run mismatch")
    require(metadata["step"] == budget.max_steps, "checkpoint step mismatch")
    require((checkpoint / "model/model.pt").is_file(), "final weights are missing")
    return {
        "status": "passed",
        "phase": phase,
        "run_id": run.name,
        "config_sha256": digest,
        "global_step": budget.max_steps,
        "target_token_coverage": coverage,
        "source_clean": runtime["code"]["dirty"] is False,
        "baseline": {key: baseline[key] for key in keys},
        "final": {key: final[key] for key in keys},
        "scope": "pretrain-dev-only; not an SFT or test gate",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/verify_a10_base.py", description=__doc__
    )
    parser.add_argument("--run", type=Path, required=True, help="completed A10 run")
    parser.add_argument("--phase", choices=PIPELINES, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    try:
        report = verify(run, phase=args.phase, root=root)
        RunArtifacts(run_id=run.name, path=run).write_json(
            "a10-development-gate.json", report
        )
    except (LLMLabError, OSError, ValueError, KeyError, StopIteration) as exc:
        print(f"error: A10 development gate failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
