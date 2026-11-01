"""在 A10 真数据 smoke 上验证受控中断与恢复。

Compare a completed A10 smoke with interruption and same-budget resume.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from _project_path import add_project_src_to_path

add_project_src_to_path()

from verify_a10_base import verify

from llm_lifecycle_lab.artifacts import RunArtifacts
from llm_lifecycle_lab.config import load_run_config
from llm_lifecycle_lab.exceptions import ContractError, LLMLabError
from llm_lifecycle_lab.training.pretrain import run_native_pretraining


class PlannedInterruption(RuntimeError):
    """Stop after update five, retaining the complete step-four checkpoint."""


def interrupt_after_fifth_update(metric: Mapping[str, Any]) -> None:
    if metric.get("step") == 5 and "train_loss" in metric:
        raise PlannedInterruption("A10 controlled interruption after checkpoint four")


def state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(state_equal(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(state_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/a10_resume_experiment.py", description=__doc__
    )
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config = load_run_config(root / "configs/pipelines/native-60m-a10-base-smoke.yaml")
    run_path = root / config.output_dir / args.run_id
    try:
        verify(args.baseline_run.resolve(), phase="smoke", root=root)
        if run_path.exists():
            raise ContractError("resume experiment run already exists; use a new ID")
        try:
            run_native_pretraining(
                config,
                run_id=args.run_id,
                workdir=root,
                metric_callback=interrupt_after_fifth_update,
            )
        except PlannedInterruption:
            pass
        else:
            raise ContractError("controlled interruption did not occur")
        latest = json.loads((run_path / "latest_checkpoint.json").read_text())
        if latest["step"] != 4:
            raise ContractError("expected the complete step-four checkpoint")
        resumed = run_native_pretraining(config, resume_run=args.run_id, workdir=root)
        verify(run_path, phase="smoke", root=root)
        left = args.baseline_run.resolve() / "checkpoints/step-00000008"
        right = Path(resumed.result.final_checkpoint)
        checks = {}
        for name, file in (
            ("weights_equal", "model/model.pt"),
            ("optimizer_scheduler_rng_equal", "optimizer_state.pt"),
        ):
            checks[name] = state_equal(
                torch.load(left / file, map_location="cpu", weights_only=True),
                torch.load(right / file, map_location="cpu", weights_only=True),
            )
        checks["trainer_and_stream_equal"] = state_equal(
            json.loads((left / "trainer_state.json").read_text()),
            json.loads((right / "trainer_state.json").read_text()),
        )
        report = {
            "status": "passed" if all(checks.values()) else "failed",
            "baseline_run": args.baseline_run.resolve().name,
            "run_id": args.run_id,
            "restored_checkpoint_step": latest["step"],
            "final_step": resumed.result.global_step,
            "tokens_seen": resumed.result.tokens_seen,
            **checks,
            "scope": "same A10, BF16, data and config; not cross-device equivalence",
        }
        RunArtifacts(run_id=args.run_id, path=run_path).write_json(
            "a10-resume-comparison.json", report
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
