from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from llm_lifecycle_lab.doctor.result import CheckStatus
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.sft_initialization import _Policy, _policy_check

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("dirty", "deviation", "expected"),
    [
        (False, False, CheckStatus.PASS),
        (True, True, CheckStatus.PASS),
        (True, False, CheckStatus.FAIL),
        (False, True, CheckStatus.FAIL),
    ],
)
def test_source_provenance_policy_matches_actual_source(dirty, deviation, expected):
    policy = _Policy(
        initialization_mode="weights-only-new-stage",
        inherit_optimizer_state=False,
        initial_global_step=0,
        require_clean_git=True,
        accepted_source_dirty=dirty,
        require_source_deviation_record=deviation,
    )
    assert _policy_check(policy).status is expected


def test_balanced_weights_cover_only_observed_strata(monkeypatch):
    script = load_script("prepare_a10_sft", monkeypatch)
    strata = {
        f"{task}.{language}": {}
        for task in ("general", "short_qa", "numeric", "structured")
        for language in ("en", "zh")
    }
    strata["classification.en"] = {}
    weights = script.balanced_stratum_weights(strata)
    assert set(weights) == set(strata)
    assert "classification.zh" not in weights
    for language in ("en", "zh"):
        assert sum(
            v for k, v in weights.items() if k.endswith(f".{language}")
        ) == pytest.approx(0.5)


def test_freeze_refuses_silent_configuration_drift(tmp_path, monkeypatch):
    script = load_script("prepare_a10_sft", monkeypatch)
    path = tmp_path / "frozen.yaml"
    script.freeze_yaml(path, {"identity": "a10"})
    script.freeze_yaml(path, {"identity": "a10"})
    with pytest.raises(ContractError, match="use a new version"):
        script.freeze_yaml(path, {"identity": "different"})


def test_launcher_rejects_cpu_before_training(monkeypatch):
    script = load_script("train_a10_sft", monkeypatch)
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ContractError, match="exactly one CUDA A10"):
        script.check_hardware()


def test_launcher_check_only_does_not_train(monkeypatch):
    script = load_script("train_a10_sft", monkeypatch)
    monkeypatch.setattr(script, "check_clean_git", lambda _root: None)
    monkeypatch.setattr(script, "check_hardware", lambda: "NVIDIA A10")
    monkeypatch.setattr(script, "_verify_gates", lambda _config, _root: {})
    monkeypatch.setattr(
        script,
        "run_native_sft",
        lambda *a, **kw: pytest.fail("check-only must not start SFT"),
    )
    assert script.main(["--phase", "smoke", "--check-only"]) == 0


def test_launcher_formal_requires_development_evidence(tmp_path, monkeypatch):
    script = load_script("train_a10_sft", monkeypatch)
    with pytest.raises(ContractError, match="complete and inspect smoke"):
        script.check_development(tmp_path, None)


def test_refreeze_is_blocked_after_first_run(tmp_path, monkeypatch):
    script = load_script("prepare_a10_sft", monkeypatch)
    (tmp_path / "runs" / "native-60m-a10-sft-v1-smoke-s42").mkdir(parents=True)
    with pytest.raises(ContractError, match="after an A10 SFT run exists"):
        script.freeze_pipelines(tmp_path, {}, {}, refreeze=True)


def test_sealed_test_claim_is_exclusive(tmp_path, monkeypatch):
    script = load_script("eval_a10_sft", monkeypatch)
    script.claim_sealed_test(tmp_path, "a" * 64)
    with pytest.raises(FileExistsError):
        script.claim_sealed_test(tmp_path, "a" * 64)


def test_sealed_test_existing_result_cannot_be_reopened(tmp_path, monkeypatch):
    script = load_script("eval_a10_sft", monkeypatch)
    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    (evaluations / "sft-test-step-00009379.json").write_text("{}")
    with pytest.raises(ContractError, match="already been evaluated"):
        script.claim_sealed_test(tmp_path, "a" * 64)


def test_manifest_rehydration_preserves_hash_identity(tmp_path, monkeypatch):
    script = load_script("prepare_a10_sft", monkeypatch)
    path = tmp_path / "manifest.json"
    snapshot = tmp_path / "snapshot.json"
    initial = {"created_at": "original", "source": {"path": "/old", "sha256": "a"}}
    path.write_text(json.dumps(initial))
    script.restore_manifest_identity(path, snapshot)
    path.write_text(
        json.dumps({"created_at": "new", "source": {"path": "/new", "sha256": "a"}})
    )
    script.restore_manifest_identity(path, snapshot)
    assert path.read_bytes() == snapshot.read_bytes()


def test_manifest_rehydration_rejects_content_tampering(tmp_path, monkeypatch):
    script = load_script("prepare_a10_sft", monkeypatch)
    path = tmp_path / "manifest.json"
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"source": {"sha256": "a"}}))
    path.write_text(json.dumps({"source": {"sha256": "b"}}))
    with pytest.raises(ContractError, match="content differs"):
        script.restore_manifest_identity(path, snapshot)
