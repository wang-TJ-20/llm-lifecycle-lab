from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAMES = (
    "data",
    "doctor",
    "eval_pretrain",
    "generate_pretrain",
    "inspect_model",
    "inspect_tokenizer",
    "model_experiment",
    "plot_training_curves",
    "pretrain_experiment",
    "train_pretrain",
    "train_tokenizer",
    "validate_config",
    "verify_reference",
)


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_script_contains_its_own_main_flow(name: str) -> None:
    source = (PROJECT_ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
    assert "def build_parser(" in source
    assert "def main(" in source


def test_project_has_no_generic_cli_entry() -> None:
    assert not (PROJECT_ROOT / "src/llm_lifecycle_lab/cli.py").exists()
    assert not (PROJECT_ROOT / "src/llm_lifecycle_lab/__main__.py").exists()
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" not in pyproject


def run_script(
    name: str,
    *arguments: str,
    cwd: Path,
    expected_code: int = 0,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / f"{name}.py"), *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_script_help_uses_its_own_name(name: str, tmp_path: Path) -> None:
    result = run_script(name, "--help", cwd=tmp_path)
    assert f"usage: python scripts/{name}.py" in result.stdout


@pytest.mark.parametrize("mode", ("train", "evaluate", "resume"))
def test_pretrain_experiment_is_standalone_and_temporary(
    mode: str, tmp_path: Path
) -> None:
    report = json.loads(
        run_script("pretrain_experiment", "--mode", mode, cwd=tmp_path).stdout
    )
    assert list(tmp_path.iterdir()) == []
    if mode == "train":
        assert report["status"] == "completed"
        assert report["steps"] == 3
        assert report["tokens_seen"] == sum(report["tokens_per_step"]) == 180
        assert report["learning_rates"] == [0.001, 0.000775, 0.000325]
        assert len(report["train_losses"]) == 3
    elif mode == "evaluate":
        assert report["sample_count"] == 4
        assert report["eval_tokens"] == report["en_tokens"] + report["zh_tokens"]
        assert report["en_tokens"] > 0 and report["zh_tokens"] > 0
        assert report["language_weighted_loss_matches"]
        assert report["standalone_eval_matches"]
        assert len(report["generated_token_ids"]) == 4
    else:
        assert report["restored_checkpoint_step"] == 1
        assert report["final_step"] == 3
        assert report["logged_train_steps"] == [1, 2, 2, 3]
        for key in (
            "weights_equal",
            "optimizer_scheduler_rng_equal",
            "trainer_and_stream_equal",
            "changed_budget_rejected",
            "completed_resume_rejected",
        ):
            assert report[key]


def test_script_error_codes_and_nested_help(tmp_path: Path) -> None:
    result = run_script("data", "fetch", "--help", cwd=tmp_path)
    assert "usage: python scripts/data.py fetch" in result.stdout
    run_script("data", "fetch", "--recipe", "missing", cwd=tmp_path, expected_code=2)
    result = run_script(
        "train_pretrain",
        "--config",
        "missing.yaml",
        cwd=tmp_path,
        expected_code=2,
    )
    assert "config file does not exist" in result.stderr
    assert not (tmp_path / "runs").exists()
    result = run_script(
        "generate_pretrain",
        "--config",
        "missing.yaml",
        "--checkpoint",
        "missing",
        cwd=tmp_path,
        expected_code=2,
    )
    assert "config file does not exist" in result.stderr


def test_plot_training_curves_writes_available_series(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {
            "step": 1,
            "train_loss": 4.0,
            "learning_rate": 0.001,
            "cuda_max_memory_allocated_bytes": 1024**3,
        },
        {
            "step": 2,
            "train_loss": 3.0,
            "learning_rate": 0.0005,
            "cuda_max_memory_allocated_bytes": 2 * 1024**3,
        },
    ]
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = run_script(
        "plot_training_curves",
        "--run",
        str(run),
        "--output",
        "charts",
        cwd=tmp_path,
    )

    report = json.loads(result.stdout)
    assert {Path(value).name for value in report["charts"]} == {
        "loss.svg",
        "learning_rate.svg",
        "memory.svg",
    }
    assert "<svg" in (tmp_path / "charts/loss.svg").read_text(encoding="utf-8")


@pytest.fixture
def mismatched_reference(tmp_path: Path) -> Path:
    pipeline = yaml.safe_load(
        (PROJECT_ROOT / "configs/pipelines/native-60m-reference.yaml").read_text(
            encoding="utf-8"
        )
    )
    pipeline["model"]["config"] = str(PROJECT_ROOT / "configs/models/tiny-60m.yaml")
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    spec = yaml.safe_load(
        (PROJECT_ROOT / "configs/reference/native-60m-pretrain-v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    spec["pipeline_config"] = "pipeline.yaml"
    spec["freeze"] = {
        "execution_sha256": "0" * 64,
        "source_sha256": "0" * 64,
        "packed_manifest_sha256": "0" * 64,
        "python_major_minor": "3.11",
        "packages": {"tokenizers": "0.21.4"},
    }
    path = tmp_path / "reference.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["inputs-only", "preflight"])
def test_reference_script_rejects_drift_before_loading_data(
    tmp_path: Path, mismatched_reference: Path, mode: str
) -> None:
    result = run_script(
        "verify_reference",
        "--spec",
        str(mismatched_reference),
        f"--{mode}",
        "--json",
        cwd=tmp_path,
        expected_code=1,
    )
    report = json.loads(result.stdout)
    assert report["scope"] == mode
    assert report["ok"] is False
    assert report["counts"]["fail"] == 1
    assert report["checks"][0]["name"] == "execution-lock"
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "arguments",
    [(), ("--inputs-only", "--preflight"), ("--run", "missing", "--inputs-only")],
)
def test_reference_script_requires_exactly_one_mode(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    run_script(
        "verify_reference",
        "--spec",
        "missing.yaml",
        *arguments,
        cwd=tmp_path,
        expected_code=2,
    )


def test_reference_script_reports_missing_completed_run(
    tmp_path: Path, mismatched_reference: Path
) -> None:
    result = run_script(
        "verify_reference",
        "--spec",
        str(mismatched_reference),
        "--run",
        "runs/missing",
        "--json",
        cwd=tmp_path,
        expected_code=1,
    )
    report = json.loads(result.stdout)
    assert report["scope"] == "completed-run"
    assert report["checks"][0]["name"] == "required-artifacts"
    assert report["checks"][0]["status"] == "fail"


@pytest.mark.parametrize("run_option", ["--run-id", "--resume-run"])
def test_training_reference_gate_does_not_create_a_run(
    tmp_path: Path, mismatched_reference: Path, run_option: str
) -> None:
    result = run_script(
        "train_pretrain",
        "--config",
        "pipeline.yaml",
        "--reference-spec",
        str(mismatched_reference),
        run_option,
        "rejected",
        cwd=tmp_path,
        expected_code=2,
    )
    assert "reference preflight failed: execution-lock" in result.stderr
    assert not (tmp_path / "runs").exists()


def test_model_experiment_runs_without_data_or_installation(tmp_path: Path) -> None:
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "model_id": "experiment-micro",
                "vocab_size": 64,
                "num_hidden_layers": 2,
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 64,
                "max_sequence_length": 32,
            }
        ),
        encoding="utf-8",
    )
    result = run_script(
        "model_experiment", "--config", str(model_path), "--json", cwd=tmp_path
    )
    report = json.loads(result.stdout)
    assert report["input_shape"] == [2, 16]
    assert report["logits_shape"] == [2, 16, 64]
    assert report["supervised_tokens"] == 30
    assert report["weight_update_max"] > 0
    assert report["grad_norm"] > 0
    assert report["cache_matches_full"]
    assert report["checkpoint_matches_full"]
    assert report["cache_key_shape"] == [2, 2, 16, 8]
    assert report["generated_shape"] == [2, 20]
    assert list(tmp_path.iterdir()) == [model_path]
    invalid = run_script(
        "model_experiment",
        "--config",
        str(model_path),
        "--sequence-length",
        "32",
        cwd=tmp_path,
        expected_code=2,
    )
    assert "sequence-length" in invalid.stderr


def test_python_scripts_prepare_train_and_evaluate(tmp_path: Path) -> None:
    records = [
        {
            "id": f"doc-{index}",
            "text": f"Training example number {index} with enough tokens for BPE.",
            "source": "script-integration",
            "language": "en" if index % 2 == 0 else "zh",
        }
        for index in range(60)
    ]
    (tmp_path / "source.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    run_script(
        "data",
        "validate",
        "--input",
        "source.jsonl",
        "--kind",
        "pretrain",
        cwd=tmp_path,
    )
    run_script(
        "data",
        "prepare",
        "--input",
        "source.jsonl",
        "--output",
        "prepared",
        "--dataset-id",
        "script-fixture",
        "--kind",
        "pretrain",
        "--license",
        "test-only",
        "--seed",
        "3",
        cwd=tmp_path,
    )
    run_script(
        "train_tokenizer",
        "--manifest",
        "prepared/data_manifest.json",
        "--output",
        "tokenizer",
        "--tokenizer-id",
        "script-bpe",
        "--vocab-size",
        "320",
        "--min-frequency",
        "1",
        cwd=tmp_path,
    )
    run_script(
        "inspect_tokenizer",
        "tokenizer",
        "--text",
        "Training example.",
        cwd=tmp_path,
    )
    run_script(
        "data",
        "pack",
        "--manifest",
        "prepared/data_manifest.json",
        "--tokenizer",
        "tokenizer",
        "--output",
        "packed",
        "--sequence-length",
        "16",
        cwd=tmp_path,
    )
    vocab_size = json.loads(
        (tmp_path / "tokenizer/tokenizer_manifest.json").read_text(encoding="utf-8")
    )["vocab_size"]
    (tmp_path / "model.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "model_route": "native",
                "provider": "native",
                "model_id": "script-micro",
                "vocab_size": vocab_size,
                "num_hidden_layers": 1,
                "hidden_size": 24,
                "num_attention_heads": 3,
                "num_key_value_heads": 1,
                "intermediate_size": 48,
                "max_sequence_length": 16,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "model_route": "native",
        "run_profile": "smoke",
        "stage": "pretrain",
        "seed": 11,
        "output_dir": "runs",
        "model": {
            "provider": "native",
            "model_id": "script-micro",
            "config": "model.yaml",
            "tokenizer": "tokenizer",
        },
        "data": {
            "manifest": "prepared/data_manifest.json",
            "packed_manifest": "packed/packed_manifest.json",
        },
        "training": {
            "device": "cpu",
            "dtype": "float32",
            "sequence_length": 16,
            "micro_batch_size": 2,
            "max_steps": 1,
            "eval_batches": 1,
        },
    }
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    run_script("inspect_model", "--config", "model.yaml", cwd=tmp_path)
    run_script("validate_config", "pipeline.yaml", cwd=tmp_path)
    report = run_script("doctor", "--config", "pipeline.yaml", "--json", cwd=tmp_path)
    assert json.loads(report.stdout)["counts"]["fail"] == 0
    run_script(
        "train_pretrain",
        "--config",
        "pipeline.yaml",
        "--run-id",
        "script-run",
        cwd=tmp_path,
    )
    report = run_script(
        "eval_pretrain",
        "--config",
        "pipeline.yaml",
        "--checkpoint",
        "runs/script-run/checkpoints/step-00000001",
        "--json",
        cwd=tmp_path,
    )
    evaluation = json.loads(report.stdout)
    assert evaluation["checkpoint_step"] == 1
    assert evaluation["eval_en_tokens"] > 0
    assert evaluation["eval_zh_tokens"] > 0
    assert Path(evaluation["report_path"]).is_file()
    saved_config = yaml.safe_load(
        (tmp_path / "runs/script-run/resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert saved_config["training"] == config["training"]
    assert saved_config["data"] == config["data"]
    result = run_script(
        "train_pretrain",
        "--config",
        "pipeline.yaml",
        "--resume-run",
        "script-run",
        cwd=tmp_path,
        expected_code=2,
    )
    assert "already reached or exceeded max_steps" in result.stderr
