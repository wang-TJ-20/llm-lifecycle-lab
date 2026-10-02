from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_lifecycle_lab.config import config_sha256, load_run_config
from llm_lifecycle_lab.contracts import (
    CheckpointMetadata,
    EvaluationReport,
    ModelRoute,
    RunManifest,
    RunProfile,
    Stage,
)
from llm_lifecycle_lab.doctor.result import CheckStatus
from llm_lifecycle_lab.exceptions import ConfigError
from llm_lifecycle_lab.sft_run_gate import SFTRunGateReport, verify_sft_run

RUN_ID = "native-60m-sft-v1-fixture"
STEP = 14_020
TARGET_TOKENS = 8_000_000
TOKENS_SEEN = 8_004_634


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _evaluation_metrics(loss: float) -> dict[str, float]:
    en_loss = loss + 1.2
    zh_loss = loss - 0.8
    return {
        "eval_loss": loss,
        "eval_perplexity": 50.0,
        "eval_tokens": 100.0,
        "eval_en_loss": en_loss,
        "eval_en_perplexity": 70.0,
        "eval_en_tokens": 40.0,
        "eval_zh_loss": zh_loss,
        "eval_zh_perplexity": 40.0,
        "eval_zh_tokens": 60.0,
    }


def _write_evaluation(run: Path, suite: str, examples: int) -> None:
    report = EvaluationReport(
        run_id=RUN_ID,
        suite=suite,
        metrics=_evaluation_metrics(3.8),
        sample_count=examples,
    )
    _write_json(
        run / "evaluations" / f"{suite}-step-{STEP:08d}.json",
        report.to_dict(),
    )


def _fixture(tmp_path: Path, *, with_test: bool = False) -> tuple[Path, Path]:
    base_gate = tmp_path / "base-gate.yaml"
    data_gate = tmp_path / "data-gate.yaml"
    suite = tmp_path / "suite.yaml"
    base_gate.write_text("gate: base\n", encoding="utf-8")
    data_gate.write_text("gate: data\n", encoding="utf-8")
    suite.write_text("suite: frozen\n", encoding="utf-8")

    pipeline = tmp_path / "pipeline.yaml"
    pipeline_value = {
        "schema_version": "1.0",
        "model_route": "native",
        "run_profile": "reproduce",
        "stage": "sft",
        "seed": 42,
        "output_dir": "runs",
        "model": {
            "provider": "native",
            "model_id": "fixture",
            "base_package": "base",
            "tokenizer": "tokenizer",
        },
        "data": {
            "base_gate": base_gate.name,
            "data_gate": data_gate.name,
            "manifest": "data_manifest.json",
            "evaluation_suite": suite.name,
            "evaluation_suite_sha256": _sha(suite),
        },
        "training": {
            "device": "cuda",
            "dtype": "bfloat16",
            "sequence_length": 32,
            "micro_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "max_train_tokens": TARGET_TOKENS,
        },
    }
    pipeline.write_text(
        yaml.safe_dump(pipeline_value, sort_keys=False),
        encoding="utf-8",
    )
    pipeline_hash = config_sha256(load_run_config(pipeline))

    tokenizer = tmp_path / "tokenizer.json"
    model_config = tmp_path / "model-config.json"
    tokenizer.write_text("fixture tokenizer\n", encoding="utf-8")
    model_config.write_text('{"model_id":"fixture"}\n', encoding="utf-8")
    tokenizer_hash = _sha(tokenizer)
    model_config_hash = _sha(model_config)
    semantic_suite_hash = "a" * 64
    parent_weights_hash = "b" * 64
    data_manifest_hash = "c" * 64

    spec_value = {
        "schema_version": "1.0",
        "gate_id": "fixture-sft-run",
        "locks": {
            "pipeline_config": pipeline.name,
            "pipeline_file_sha256": _sha(pipeline),
            "pipeline_config_sha256": pipeline_hash,
            "base_gate": base_gate.name,
            "base_gate_sha256": _sha(base_gate),
            "data_gate": data_gate.name,
            "data_gate_sha256": _sha(data_gate),
            "evaluation_suite": suite.name,
            "evaluation_suite_file_sha256": _sha(suite),
            "evaluation_suite_sha256": semantic_suite_hash,
        },
        "identity": {
            "model_route": "native",
            "run_profile": "reproduce",
            "stage": "sft",
            "parent_run_id": "base-fixture",
            "parent_checkpoint_step": 45_191,
            "parent_weights_sha256": parent_weights_hash,
            "parent_model_config_sha256": model_config_hash,
            "tokenizer_sha256": tokenizer_hash,
            "data_manifest_sha256": data_manifest_hash,
        },
        "training": {
            "budget_mode": "max_train_tokens",
            "global_step": STEP,
            "target_train_tokens": TARGET_TOKENS,
            "expected_tokens_seen": TOKENS_SEEN,
            "minimum_target_token_coverage": 1.0,
            "maximum_target_token_coverage": 1.001,
            "required_training_metric_keys": [
                "train_loss",
                "gradient_norm",
                "eval_loss",
                "eval_en_loss",
                "eval_zh_loss",
                "eval_tokens",
                "eval_en_tokens",
                "eval_zh_tokens",
            ],
        },
        "evaluation": {
            "dev_suite": "sft-dev",
            "dev_examples": 1459,
            "test_suite": "sft-test",
            "test_examples": 1458,
            "required_metric_keys": list(_evaluation_metrics(3.8)),
        },
        "runtime": {
            "platform_system": "Linux",
            "accelerator_type": "cuda",
            "require_clean_source": True,
        },
    }
    spec = tmp_path / "run-gate.yaml"
    spec.write_text(
        yaml.safe_dump(spec_value, sort_keys=False),
        encoding="utf-8",
    )

    run = tmp_path / "runs" / RUN_ID
    run.mkdir(parents=True)
    manifest = RunManifest(
        run_id=RUN_ID,
        config_sha256=pipeline_hash,
        model_route=ModelRoute.NATIVE,
        run_profile=RunProfile.REPRODUCE,
        stage=Stage.SFT,
        status="completed",
    )
    _write_json(run / "run_manifest.json", manifest.to_dict())
    (run / "resolved_config.yaml").write_bytes(pipeline.read_bytes())
    _write_json(
        run / "preflight.json",
        {
            "scope": "preflight",
            "git_checked": True,
            "base_gate": {
                "gate_id": "native-60m-sft-v1-base-init",
                "scope": "inputs-only",
                "ok": True,
            },
            "data_gate": {
                "gate_id": "native-60m-sft-v1-data",
                "scope": "preflight",
                "ok": True,
            },
            "evaluation_suite": {
                "path": str(suite.resolve()),
                "sha256": _sha(suite),
            },
        },
    )
    _write_json(
        run / "initialization.json",
        {
            "mode": "weights-only-new-stage",
            "optimizer_inherited": False,
            "initial_step": 0,
            "parent_checkpoint": {
                "run_id": "base-fixture",
                "step": 45_191,
                "model_route": "native",
                "stage": "pretrain",
                "tokenizer_sha256": tokenizer_hash,
            },
            "parent_weights_sha256": parent_weights_hash,
            "parent_model_config_sha256": model_config_hash,
            "data_manifest_sha256": data_manifest_hash,
            "evaluation_suite_sha256": semantic_suite_hash,
            "sampling": {"strategy": "supervised-token-quota"},
        },
    )
    _write_json(run / "sft_data_summary.json", {"splits": {}})
    _write_json(
        run / "training_budget.json",
        {
            "mode": "max_train_tokens",
            "max_steps": STEP,
            "target_train_tokens": TARGET_TOKENS,
            "requested_max_train_tokens": TARGET_TOKENS,
            "requested_max_steps": None,
            "requested_num_epochs": None,
        },
    )
    coverage = TOKENS_SEEN / TARGET_TOKENS
    _write_json(
        run / "training_result.json",
        {
            "global_step": STEP,
            "tokens_seen": TOKENS_SEEN,
            "target_train_tokens": TARGET_TOKENS,
            "target_token_coverage": coverage,
            "elapsed_seconds": 100.0,
            "final_loss": 2.5,
            "best_eval_loss": 3.8,
            "final_checkpoint": str(
                (run / "checkpoints" / f"step-{STEP:08d}").resolve()
            ),
        },
    )
    baseline = {"event": "baseline", "step": 0, **_evaluation_metrics(4.8)}
    final = {
        "step": STEP,
        "train_loss": 2.5,
        "gradient_norm": 1.0,
        **_evaluation_metrics(3.8),
    }
    (run / "metrics.jsonl").write_text(
        json.dumps(baseline) + "\n" + json.dumps(final) + "\n",
        encoding="utf-8",
    )
    _write_json(
        run / "runtime_environment.json",
        {
            "platform": {"system": "Linux"},
            "accelerator": {"type": "cuda"},
            "code": {"dirty": False, "status_entries": 0},
        },
    )
    _write_json(run / "model_config.json", {"model_id": "fixture"})
    (run / "tokenizer").mkdir()
    (run / "tokenizer/tokenizer.json").write_bytes(tokenizer.read_bytes())
    _write_json(run / "tokenizer/tokenizer_manifest.json", {"fixture": True})

    checkpoint = run / "checkpoints" / f"step-{STEP:08d}"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model/model.pt").write_bytes(b"weights")
    (checkpoint / "model/config.json").write_bytes(model_config.read_bytes())
    (checkpoint / "optimizer_state.pt").write_bytes(b"optimizer")
    _write_json(
        checkpoint / "trainer_state.json",
        {"global_step": STEP, "tokens_seen": TOKENS_SEEN},
    )
    metadata = CheckpointMetadata(
        checkpoint_id=f"step-{STEP:08d}",
        run_id=RUN_ID,
        model_route=ModelRoute.NATIVE,
        stage=Stage.SFT,
        step=STEP,
        tokenizer_sha256=tokenizer_hash,
        config_sha256=pipeline_hash,
    )
    _write_json(
        checkpoint / "checkpoint_metadata.json",
        metadata.to_dict(),
    )
    _write_json(
        run / "latest_checkpoint.json",
        {
            "checkpoint_id": f"step-{STEP:08d}",
            "path": f"checkpoints/step-{STEP:08d}",
            "step": STEP,
        },
    )
    _write_evaluation(run, "sft-dev", 1459)
    if with_test:
        _write_evaluation(run, "sft-test", 1458)
    return spec, run


def _statuses(report: SFTRunGateReport) -> dict[str, CheckStatus]:
    return {check.name: check.status for check in report.checks}


def test_completed_run_passes_development_gate(tmp_path: Path) -> None:
    spec, run = _fixture(tmp_path)

    report = verify_sft_run(spec, run, workdir=tmp_path)

    assert report.exit_code == 0
    assert report.scope == "development"
    assert report.counts == {"pass": 11, "warn": 0, "fail": 0}


def test_sealed_test_is_required_only_in_sealed_scope(tmp_path: Path) -> None:
    spec, run = _fixture(tmp_path)

    report = verify_sft_run(spec, run, workdir=tmp_path, sealed_test=True)

    assert report.exit_code == 1
    assert _statuses(report)["sealed-test-evaluation"] is CheckStatus.FAIL

    _write_evaluation(run, "sft-test", 1458)
    report = verify_sft_run(spec, run, workdir=tmp_path, sealed_test=True)
    assert report.exit_code == 0
    assert report.counts == {"pass": 12, "warn": 0, "fail": 0}


def test_token_coverage_drift_fails(tmp_path: Path) -> None:
    spec, run = _fixture(tmp_path)
    result_path = run / "training_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["tokens_seen"] += 1
    result["target_token_coverage"] = result["tokens_seen"] / TARGET_TOKENS
    _write_json(result_path, result)

    report = verify_sft_run(spec, run, workdir=tmp_path)

    assert _statuses(report)["training-result"] is CheckStatus.FAIL
    assert _statuses(report)["final-checkpoint"] is CheckStatus.FAIL


def test_spec_rejects_unknown_fields(tmp_path: Path) -> None:
    spec, run = _fixture(tmp_path)
    value = yaml.safe_load(spec.read_text(encoding="utf-8"))
    value["unregistered_threshold"] = 1
    spec.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown unregistered_threshold"):
        verify_sft_run(spec, run, workdir=tmp_path)
