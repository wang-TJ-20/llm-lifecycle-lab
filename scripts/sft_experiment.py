"""离线验证微型 Base→SFT→同协议评测和恢复；不作为语言能力结论。

Verify a tiny offline Base-to-SFT workflow, comparable evaluation, and exact resume.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from _project_path import add_project_src_to_path

add_project_src_to_path()

from pretrain_experiment import (
    PlannedInterruption,
    assert_state_equal,
    interrupt_after_second_update,
    prepare_experiment,
)

from llm_lifecycle_lab.config import dump_yaml
from llm_lifecycle_lab.contracts import RunConfig, Stage
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.prepare import prepare_dataset
from llm_lifecycle_lab.evaluation.runner import evaluate_capabilities
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.model.native import load_native_model_config
from llm_lifecycle_lab.training.pretrain import run_native_pretraining
from llm_lifecycle_lab.training.sft import run_native_sft

ROOT = Path(__file__).resolve().parents[1]


def prepare_sft_fixture(root: Path) -> Path:
    rows = [
        {
            "id": f"sft-{language}-{index}",
            "source_id": f"sft-example-{index}",
            "language": language,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": str(index)},
            ],
        }
        for index in range(80)
        for language, prompt in (
            ("en", f"Write this number again using digits only: {index}."),
            ("zh", f"请重新写出这个数字，只输出数字：{index}。"),
        )
    ]
    source = root / "sft-source.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    target = root / "sft-data"
    prepare_dataset(
        source,
        target,
        dataset_id="sft-mechanism-fixture",
        record_kind="sft",
        license_name="Apache-2.0",
        group_by="source_id",
        seed=42,
    )
    return target / "data_manifest.json"


def run_experiment(root: Path, *, mode: str) -> dict[str, Any]:
    config = prepare_experiment(root)
    model_path = Path(config.model["config"])
    model = replace(load_native_model_config(model_path), max_sequence_length=512)
    model_path.write_text(dump_yaml(asdict(model)), encoding="utf-8")
    base = run_native_pretraining(config, run_id="base", workdir=ROOT)
    sft_config: RunConfig = replace(
        config,
        stage=Stage.SFT,
        model={
            "provider": "native",
            "model_id": model.model_id,
            "init_checkpoint": base.result.final_checkpoint,
            "tokenizer": config.model["tokenizer"],
        },
        data={
            "manifest": str(prepare_sft_fixture(root)),
            "evaluation_suite": str(ROOT / "configs/evaluation/lifecycle-v1.yaml"),
        },
        training={**config.training, "sequence_length": 512},
    )
    before_dir = root / "base-evaluation"
    before = evaluate_capabilities(
        checkpoint=base.result.final_checkpoint,
        tokenizer_dir=config.model["tokenizer"],
        suite_path=sft_config.data["evaluation_suite"],
        output=before_dir,
        max_new_tokens=4,
        prompt_protocol="native-chat-v1",
    )
    sft = run_native_sft(sft_config, run_id="sft", workdir=ROOT)
    after = evaluate_capabilities(
        checkpoint=sft.result.final_checkpoint,
        tokenizer_dir=config.model["tokenizer"],
        suite_path=sft_config.data["evaluation_suite"],
        output=root / "sft-evaluation",
        max_new_tokens=4,
        prompt_protocol="native-chat-v1",
        baseline_path=before_dir / "report.json",
    )
    result = {
        "base_steps": base.result.global_step,
        "sft_steps": sft.result.global_step,
        "sft_supervised_tokens": sft.result.tokens_seen,
        "parent_weights_preserved": sha256_file(
            Path(base.result.final_checkpoint) / "model/model.pt"
        )
        == before["model"]["weights_sha256"],
        "sft_weights_updated": before["model"]["weights_sha256"]
        != after["model"]["weights_sha256"],
        "same_evaluation_protocol": before["protocol_sha256"]
        == after["protocol_sha256"],
        "compared_metrics": len(after["comparison"]["metrics"]),
        "note": "Synthetic mechanism test only; no held-out language capability claim.",
    }
    if mode == "resume":
        try:
            run_native_sft(
                sft_config,
                run_id="interrupted",
                workdir=ROOT,
                metric_callback=interrupt_after_second_update,
            )
        except PlannedInterruption:
            pass
        else:
            raise AssertionError("planned SFT interruption did not occur")
        resumed = run_native_sft(
            sft_config,
            resume_run="interrupted",
            workdir=ROOT,
        )
        left, right = (
            Path(sft.result.final_checkpoint),
            Path(resumed.result.final_checkpoint),
        )
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
        prog="python scripts/sft_experiment.py", description=__doc__
    )
    parser.add_argument("--mode", choices=("train", "resume"), default="train")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)
    try:
        with TemporaryDirectory(prefix="llmlab-sft-") as temporary:
            result = run_experiment(Path(temporary), mode=args.mode)
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
