from __future__ import annotations

import json
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
    "inspect_model",
    "inspect_tokenizer",
    "train_pretrain",
    "train_tokenizer",
    "validate_config",
    "verify_reference",
)


def run_script(
    name: str,
    *arguments: str,
    cwd: Path,
    expected_code: int = 0,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / f"{name}.py"), *arguments],
        cwd=cwd,
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


def test_python_and_legacy_entries_list_identical_recipes(tmp_path: Path) -> None:
    scripted = run_script("data", "recipes", "--json", cwd=tmp_path)
    legacy = subprocess.run(
        [sys.executable, "-m", "llm_lifecycle_lab", "data", "recipes", "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(scripted.stdout) == json.loads(legacy.stdout)


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
                "model_route": "native-smoke",
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
        "model_route": "native-smoke",
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
