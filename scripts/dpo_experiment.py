"""离线验证微型 SFT→DPO、冻结 reference、计权与精确恢复。

Verify tiny offline SFT-to-DPO mechanics; this is not a preference-quality claim.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from _project_path import add_project_src_to_path

add_project_src_to_path()

from pretrain_experiment import (
    PlannedInterruption,
    assert_state_equal,
    interrupt_after_second_update,
    prepare_experiment,
)
from sft_experiment import prepare_sft_fixture

from llm_lifecycle_lab.config import dump_yaml
from llm_lifecycle_lab.contracts import RunConfig, Stage
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.prepare import prepare_dataset
from llm_lifecycle_lab.model.native import load_native_model_config
from llm_lifecycle_lab.training.dpo import run_native_dpo
from llm_lifecycle_lab.training.pretrain import run_native_pretraining
from llm_lifecycle_lab.training.sft import run_native_sft

ROOT = Path(__file__).resolve().parents[1]


def prepare_dpo_fixture(root: Path) -> Path:
    rows = [
        {
            "id": f"dpo-{language}-{index}",
            "source_id": f"preference-{index}",
            "language": language,
            "prompt": prompt,
            "chosen": str(index),
            "rejected": str((index + 7) % 80),
        }
        for index in range(80)
        for language, prompt in (
            ("en", f"Reply with the integer {index} using digits only."),
            ("zh", f"请只用数字回答整数{index}。"),
        )
    ]
    source = root / "dpo-source.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    target = root / "dpo-data"
    prepare_dataset(
        source,
        target,
        dataset_id="dpo-mechanism-fixture",
        record_kind="dpo",
        license_name="Apache-2.0",
        group_by="source_id",
        seed=42,
    )
    return target / "data_manifest.json"


def run_experiment(root: Path, *, mode: str) -> dict:
    pretrain_config = prepare_experiment(root)
    model_path = Path(pretrain_config.model["config"])
    model = replace(load_native_model_config(model_path), max_sequence_length=512)
    model_path.write_text(dump_yaml(asdict(model)), encoding="utf-8")
    base = run_native_pretraining(pretrain_config, run_id="base", workdir=ROOT)
    sft_config: RunConfig = replace(
        pretrain_config,
        stage=Stage.SFT,
        model={
            "provider": "native",
            "model_id": model.model_id,
            "init_checkpoint": base.result.final_checkpoint,
            "tokenizer": pretrain_config.model["tokenizer"],
        },
        data={
            "manifest": str(prepare_sft_fixture(root)),
            "evaluation_suite": str(ROOT / "configs/evaluation/lifecycle-v1.yaml"),
        },
        training={**pretrain_config.training, "sequence_length": 512},
    )
    sft = run_native_sft(sft_config, run_id="sft", workdir=ROOT)
    parent_hash = sha256_file(Path(sft.result.final_checkpoint) / "model/model.pt")
    dpo_config = replace(
        sft_config,
        stage=Stage.DPO,
        model={**sft_config.model, "init_checkpoint": sft.result.final_checkpoint},
        data={**sft_config.data, "manifest": str(prepare_dpo_fixture(root))},
        training={**sft_config.training, "beta": 0.1},
    )
    continuous = run_native_dpo(dpo_config, run_id="dpo", workdir=ROOT)
    metrics = [
        json.loads(line)
        for line in (continuous.artifacts.path / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    result = {
        "dpo_steps": continuous.result.global_step,
        "response_tokens_seen": continuous.result.tokens_seen,
        "sft_parent_unchanged": (
            sha256_file(Path(sft.result.final_checkpoint) / "model/model.pt")
            == parent_hash
        ),
        "frozen_reference_scores": (
            continuous.artifacts.path / "reference_scores.json"
        ).is_file(),
        "pair_weighted_loss": any("report_pair_accuracy" in row for row in metrics),
        "note": "Synthetic mechanism test only; no preference-quality claim.",
    }
    if mode == "resume":
        try:
            run_native_dpo(
                dpo_config,
                run_id="interrupted",
                workdir=ROOT,
                metric_callback=interrupt_after_second_update,
            )
        except PlannedInterruption:
            pass
        else:
            raise AssertionError("planned DPO interruption did not occur")
        resumed = run_native_dpo(dpo_config, resume_run="interrupted", workdir=ROOT)
        left = Path(continuous.result.final_checkpoint)
        right = Path(resumed.result.final_checkpoint)
        for name in ("model/model.pt", "optimizer_state.pt"):
            assert_state_equal(
                torch.load(left / name, map_location="cpu", weights_only=True),
                torch.load(right / name, map_location="cpu", weights_only=True),
            )
        assert_state_equal(
            json.loads((left / "trainer_state.json").read_text()),
            json.loads((right / "trainer_state.json").read_text()),
        )
        result["resume_weights_optimizer_scheduler_rng_equal"] = True
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/dpo_experiment.py", description=__doc__
    )
    parser.add_argument("--mode", choices=("train", "resume"), default="train")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)
    with TemporaryDirectory(prefix="llmlab-dpo-") as temporary:
        result = run_experiment(Path(temporary), mode=args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
