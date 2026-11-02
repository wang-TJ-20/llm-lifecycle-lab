from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from llm_lifecycle_lab.doctor.result import CheckStatus
from llm_lifecycle_lab.sft_run_gate import _preflight_check


@pytest.mark.parametrize(
    ("base_id", "data_id", "expected"),
    [
        ("fixture-base", "fixture-data", CheckStatus.PASS),
        ("different-base", "fixture-data", CheckStatus.FAIL),
        ("fixture-base", "different-data", CheckStatus.FAIL),
        (None, "fixture-data", CheckStatus.FAIL),
        ("fixture-base", None, CheckStatus.FAIL),
        ("", "fixture-data", CheckStatus.FAIL),
    ],
)
def test_preflight_uses_gate_ids_from_locked_yaml(
    tmp_path: Path,
    base_id: str | None,
    data_id: str | None,
    expected: CheckStatus,
) -> None:
    base_gate = tmp_path / "base.yaml"
    data_gate = tmp_path / "data.yaml"
    base_gate.write_text(yaml.safe_dump({"gate_id": base_id}), encoding="utf-8")
    data_gate.write_text(yaml.safe_dump({"gate_id": data_id}), encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    spec = SimpleNamespace(
        locks=SimpleNamespace(
            base_gate=base_gate,
            data_gate=data_gate,
            evaluation_suite=suite,
            evaluation_suite_file_sha256="a" * 64,
        )
    )
    preflight = {
        "scope": "preflight",
        "git_checked": True,
        "base_gate": {"gate_id": "fixture-base", "ok": True},
        "data_gate": {
            "gate_id": "fixture-data",
            "scope": "preflight",
            "ok": True,
        },
        "evaluation_suite": {"path": str(suite), "sha256": "a" * 64},
    }

    assert _preflight_check(spec, preflight).status is expected
