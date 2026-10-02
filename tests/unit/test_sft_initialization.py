from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

import llm_lifecycle_lab.sft_initialization as sft_initialization
from llm_lifecycle_lab.doctor.result import CheckStatus
from llm_lifecycle_lab.exceptions import ConfigError
from llm_lifecycle_lab.sft_initialization import (
    SFTInitializationReport,
    verify_sft_initialization,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = PROJECT_ROOT / "configs/gates/native-60m-sft-v1-base-init.yaml"
PARAMETER_COUNT = 62_927_616


def _write_spec(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    value = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    mutate(value)
    path = tmp_path / "sft-init.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _statuses(report: SFTInitializationReport) -> dict[str, CheckStatus]:
    return {check.name: check.status for check in report.checks}


def test_repository_sft_initialization_inputs_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sft_initialization,
        "_load_model_parameter_count",
        lambda _package, _config: PARAMETER_COUNT,
    )

    report = verify_sft_initialization(SPEC_PATH, workdir=PROJECT_ROOT)

    assert report.exit_code == 0
    assert report.scope == "inputs-only"
    assert report.counts == {"pass": 10, "warn": 0, "fail": 0}
    assert _statuses(report)["model-loadability"] is CheckStatus.PASS
    assert "clean-worktree" not in _statuses(report)


def test_artifact_drift_fails_before_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(value: dict[str, object]) -> None:
        artifacts = value["package_artifacts"]
        assert isinstance(artifacts, dict)
        model = artifacts["model.pt"]
        assert isinstance(model, dict)
        model["sha256"] = "0" * 64

    spec = _write_spec(tmp_path, mutate)

    def unexpected_load(*_args: object) -> int:
        raise AssertionError("model loading must not run after artifact drift")

    monkeypatch.setattr(
        sft_initialization, "_load_model_parameter_count", unexpected_load
    )
    report = verify_sft_initialization(spec, workdir=PROJECT_ROOT)

    assert report.exit_code == 1
    assert _statuses(report)["base-artifacts"] is CheckStatus.FAIL
    assert "model-loadability" not in _statuses(report)


def test_provenance_identity_drift_fails_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(value: dict[str, object]) -> None:
        identity = value["identity"]
        assert isinstance(identity, dict)
        identity["source_run_id"] = "different-base-run"

    spec = _write_spec(tmp_path, mutate)
    monkeypatch.setattr(
        sft_initialization,
        "_load_model_parameter_count",
        lambda _package, _config: PARAMETER_COUNT,
    )

    report = verify_sft_initialization(spec, workdir=PROJECT_ROOT)

    assert report.exit_code == 1
    statuses = _statuses(report)
    assert statuses["base-provenance"] is CheckStatus.FAIL
    assert statuses["source-evidence-contract"] is CheckStatus.FAIL
    assert "model-loadability" not in statuses


def test_preflight_rejects_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sft_initialization,
        "_load_model_parameter_count",
        lambda _package, _config: PARAMETER_COUNT,
    )

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments[0] == "rev-parse":
            return "1" * 40
        return " M src/llm_lifecycle_lab/example.py"

    monkeypatch.setattr(sft_initialization, "_run_git", fake_git)
    report = verify_sft_initialization(
        SPEC_PATH,
        workdir=PROJECT_ROOT,
        check_git=True,
    )

    assert report.exit_code == 1
    assert report.scope == "preflight"
    assert _statuses(report)["clean-worktree"] is CheckStatus.FAIL


def test_spec_rejects_unknown_fields(tmp_path: Path) -> None:
    def mutate(value: dict[str, object]) -> None:
        value["unfrozen_sft_data"] = "must not be added implicitly"

    spec = _write_spec(tmp_path, mutate)

    with pytest.raises(ConfigError, match="unknown unfrozen_sft_data"):
        verify_sft_initialization(spec, workdir=PROJECT_ROOT)
