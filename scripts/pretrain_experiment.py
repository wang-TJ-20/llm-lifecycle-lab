"""在临时目录验证预训练、评测和中断恢复，不下载数据。

Run offline pretraining, evaluation, and resume experiments in a temporary directory.
The fixture is synthetic; its metrics are not evidence of language capability.
Core workflow: llm_lifecycle_lab.training.pretrain.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
import yaml
from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.contracts import ModelRoute, RunConfig, RunProfile, Stage
from llm_lifecycle_lab.data import prepare_dataset
from llm_lifecycle_lab.data.packing import materialize_packed_pretraining_dataset
from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError, LLMLabError
from llm_lifecycle_lab.model.native import NativeTransformer, load_native_model_config
from llm_lifecycle_lab.model.protocol import GenerationConfig
from llm_lifecycle_lab.tokenizer import NativeTokenizer, train_native_tokenizer
from llm_lifecycle_lab.training.pretrain import (
    PretrainingRun,
    evaluate_native_pretraining,
    run_native_pretraining,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_experiment(root: Path) -> RunConfig:
    rows = [
        {
            "id": f"{language}-{index}",
            "source_id": f"arithmetic-{index}",
            "source": "tutorial-synthetic",
            "language": language,
            "text": text,
        }
        for index in range(40)
        for language, text in (
            (
                "en",
                f"Example {index}: {index} plus one is {index + 1}. We read numbers.",
            ),
            ("zh", f"例题{index}：{index}加一等于{index + 1}。我们学习数字。"),
        )
    ]
    source = root / "source.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    prepared = root / "prepared"
    data = prepare_dataset(
        source,
        prepared,
        dataset_id="tutorial-pretrain",
        record_kind="pretrain",
        license_name="self-written-example",
        seed=42,
        group_by="source_id",
    )
    assert all(split.records > 0 for split in data.splits)
    manifest = prepared / "data_manifest.json"
    tokenizer_path = root / "tokenizer"
    tokenizer = train_native_tokenizer(
        manifest,
        tokenizer_path,
        tokenizer_id="tutorial-bpe",
        vocab_size=320,
        min_frequency=1,
    )
    packed = root / "packed"
    materialize_packed_pretraining_dataset(
        manifest,
        tokenizer_path,
        packed,
        sequence_length=16,
    )
    model_path = root / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "model_id": "tutorial-pretrain",
                "vocab_size": tokenizer.vocab_size,
                "num_hidden_layers": 1,
                "hidden_size": 24,
                "num_attention_heads": 3,
                "num_key_value_heads": 1,
                "intermediate_size": 48,
                "max_sequence_length": 64,
            }
        ),
        encoding="utf-8",
    )
    return RunConfig(
        model_route=ModelRoute.NATIVE,
        run_profile=RunProfile.SMOKE,
        stage=Stage.PRETRAIN,
        seed=11,
        output_dir=str(root / "runs"),
        model={
            "provider": "native",
            "model_id": "tutorial-pretrain",
            "config": str(model_path),
            "tokenizer": str(tokenizer_path),
        },
        data={
            "manifest": str(manifest),
            "packed_manifest": str(packed / "packed_manifest.json"),
        },
        training={
            "device": "cpu",
            "dtype": "float32",
            "sequence_length": 16,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "max_steps": 3,
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "checkpoint_interval": 1,
            "eval_interval": 1,
            "eval_batches": 2,
            "log_interval": 1,
        },
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(run: PretrainingRun) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run.artifacts.path / "metrics.jsonl").read_text().splitlines()
    ]


def train_summary(run: PretrainingRun) -> dict[str, Any]:
    rows = read_metrics(run)
    steps = [row for row in rows if "train_loss" in row]
    assert [row["step"] for row in steps] == [1, 2, 3]
    assert rows[0]["event"] == "baseline" and rows[0]["step"] == 0
    assert read_json(run.artifacts.path / "run_manifest.json")["status"] == "completed"
    assert run.result.tokens_seen == sum(row["tokens"] for row in steps)
    assert all(math.isfinite(row["train_loss"]) for row in steps)
    assert Path(run.result.final_checkpoint, "model/model.pt").is_file()
    return {
        "parameters": run.parameter_count,
        "steps": run.result.global_step,
        "tokens_seen": run.result.tokens_seen,
        "learning_rates": [round(row["learning_rate"], 7) for row in steps],
        "tokens_per_step": [row["tokens"] for row in steps],
        "train_losses": [round(row["train_loss"], 6) for row in steps],
        "baseline_dev_loss": round(rows[0]["eval_loss"], 6),
        "final_dev_loss": round(steps[-1]["eval_loss"], 6),
        "status": "completed",
    }


def evaluation_summary(config: RunConfig, run: PretrainingRun) -> dict[str, Any]:
    report = evaluate_native_pretraining(
        config,
        checkpoint=run.result.final_checkpoint,
        split="dev",
        workdir=PROJECT_ROOT,
    )
    saved = read_json(Path(report["report_path"]))
    en_tokens, zh_tokens = report["eval_en_tokens"], report["eval_zh_tokens"]
    assert en_tokens > 0 and zh_tokens > 0
    assert report["eval_tokens"] == en_tokens + zh_tokens
    weighted = (
        report["eval_en_loss"] * en_tokens + report["eval_zh_loss"] * zh_tokens
    ) / report["eval_tokens"]
    assert math.isclose(weighted, report["eval_loss"], rel_tol=1e-6, abs_tol=1e-6)
    final_eval = read_metrics(run)[-1]["eval_loss"]
    assert math.isclose(final_eval, report["eval_loss"], rel_tol=1e-6, abs_tol=1e-6)

    model = NativeTransformer(load_native_model_config(config.model["config"]))
    model.load(Path(run.result.final_checkpoint) / "model")
    tokenizer = NativeTokenizer.from_directory(config.model["tokenizer"])
    prompt = "Example"
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)])
    generation = model.generate(
        input_ids=ids,
        attention_mask=None,
        config=GenerationConfig(max_new_tokens=4),
    )
    return {
        "sample_count": saved["sample_count"],
        "eval_tokens": int(report["eval_tokens"]),
        "en_tokens": int(en_tokens),
        "zh_tokens": int(zh_tokens),
        "eval_loss": round(report["eval_loss"], 6),
        "eval_en_loss": round(report["eval_en_loss"], 6),
        "eval_zh_loss": round(report["eval_zh_loss"], 6),
        "eval_perplexity": round(report["eval_perplexity"], 6),
        "eval_bits_per_byte": round(report["eval_bits_per_byte"], 6),
        "language_weighted_loss_matches": True,
        "standalone_eval_matches": True,
        "prompt": prompt,
        "generated_token_ids": generation.token_ids[0, ids.shape[1] :].tolist(),
        "decoded_with_specials": tokenizer.decode(
            generation.token_ids[0].tolist(),
            skip_special_tokens=False,
        ),
    }


class PlannedInterruption(RuntimeError):
    """Stop after a logged update, before that update's checkpoint is saved."""


def interrupt_after_second_update(metric: Mapping[str, Any]) -> None:
    if metric.get("step") == 2 and "train_loss" in metric:
        raise PlannedInterruption("tutorial interruption before checkpoint 2")


def assert_state_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_state_equal(first, second)
    else:
        assert left == right


def resume_summary(config: RunConfig, baseline: PretrainingRun) -> dict[str, Any]:
    try:
        run_native_pretraining(
            config,
            run_id="interrupted",
            workdir=PROJECT_ROOT,
            metric_callback=interrupt_after_second_update,
        )
    except PlannedInterruption:
        pass
    else:
        raise AssertionError("the interruption did not run")

    interrupted_path = Path(config.output_dir) / "interrupted"
    assert read_json(interrupted_path / "run_manifest.json")["status"] == "failed"
    latest = read_json(interrupted_path / "latest_checkpoint.json")
    assert latest["step"] == 1
    assert not (interrupted_path / "checkpoints/step-00000002").exists()
    resumed = run_native_pretraining(
        config,
        resume_run="interrupted",
        workdir=PROJECT_ROOT,
    )
    left, right = (
        Path(baseline.result.final_checkpoint),
        Path(resumed.result.final_checkpoint),
    )
    for filename in ("model/model.pt", "optimizer_state.pt"):
        assert_state_equal(
            torch.load(left / filename, map_location="cpu", weights_only=True),
            torch.load(right / filename, map_location="cpu", weights_only=True),
        )
    assert_state_equal(
        read_json(left / "trainer_state.json"), read_json(right / "trainer_state.json")
    )
    assert read_json(interrupted_path / "run_manifest.json")["status"] == "completed"
    rows = read_metrics(resumed)
    assert any(
        row.get("event") == "resume-baseline" and row["step"] == 1 for row in rows
    )

    changed = replace(config, training={**config.training, "max_steps": 4})
    try:
        run_native_pretraining(changed, resume_run="interrupted", workdir=PROJECT_ROOT)
    except ArtifactError as exc:
        assert "resolved config" in str(exc)
    else:
        raise AssertionError("changed resume budget was accepted")
    try:
        run_native_pretraining(config, resume_run="interrupted", workdir=PROJECT_ROOT)
    except ConfigError as exc:
        assert "reached or exceeded" in str(exc)
    else:
        raise AssertionError("completed run was extended")
    return {
        "restored_checkpoint_step": latest["step"],
        "final_step": resumed.result.global_step,
        "tokens_seen": resumed.result.tokens_seen,
        "weights_equal": True,
        "optimizer_scheduler_rng_equal": True,
        "trainer_and_stream_equal": True,
        "logged_train_steps": [row["step"] for row in rows if "train_loss" in row],
        "changed_budget_rejected": True,
        "completed_resume_rejected": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/pretrain_experiment.py",
        description="离线运行微型预训练、评测或恢复对照；所有产物自动清理。",
    )
    parser.add_argument(
        "--mode", choices=("train", "evaluate", "resume"), default="train"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        with TemporaryDirectory(prefix="llmlab-pretrain-") as temporary:
            config = prepare_experiment(Path(temporary))
            run = run_native_pretraining(
                config, run_id="baseline", workdir=PROJECT_ROOT
            )
            if args.mode == "train":
                summary = train_summary(run)
            elif args.mode == "evaluate":
                summary = evaluation_summary(config, run)
            else:
                summary = resume_summary(config, run)
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
