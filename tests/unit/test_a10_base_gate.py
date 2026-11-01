from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_lifecycle_lab.config import config_sha256, load_run_config
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.training.engine import EngineConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "a10_gate_test", PROJECT_ROOT / "scripts/verify_a10_base.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train = SimpleNamespace(name="train", examples=1000, supervised_tokens=511000)
    monkeypatch.setattr(
        module,
        "load_packed_pretraining_manifest",
        lambda _: SimpleNamespace(splits=[train]),
    )
    return module, train


def make_run(tmp_path, gate, phase="smoke"):
    module, train = gate
    config = load_run_config(
        PROJECT_ROOT / "configs/pipelines" / module.PIPELINES[phase]
    )
    _, budget = EngineConfig.from_dict(config.training).resolve_budget(
        examples_per_epoch=train.examples,
        supervised_tokens_per_epoch=train.supervised_tokens,
    )
    run = tmp_path / "a10-test"
    checkpoint = run / "checkpoints" / f"step-{budget.max_steps:08d}"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model/model.pt").touch()
    digest = config_sha256(config)
    values = {
        "run_manifest.json": {
            "run_id": run.name,
            "status": "completed",
            "config_sha256": digest,
        },
        "training_result.json": {
            "global_step": budget.max_steps,
            "tokens_seen": budget.estimated_train_tokens,
            "target_train_tokens": budget.target_train_tokens,
            "target_token_coverage": (
                budget.estimated_train_tokens / budget.target_train_tokens
            ),
        },
        "runtime_environment.json": {
            "accelerator": {
                "type": "cuda",
                "devices": [{"name": "NVIDIA A10"}],
            },
            "code": {"dirty": False, "commit": "a" * 40},
        },
        f"checkpoints/step-{budget.max_steps:08d}/checkpoint_metadata.json": {
            "config_sha256": digest,
            "run_id": run.name,
            "step": budget.max_steps,
        },
    }
    for name, value in values.items():
        (run / name).write_text(json.dumps(value), encoding="utf-8")
    rows = [
        {
            "event": "baseline",
            "step": 0,
            "eval_loss": 10.0,
            "eval_en_loss": 10.0,
            "eval_zh_loss": 10.0,
        },
        {
            "step": budget.max_steps,
            "eval_loss": 4.0,
            "eval_en_loss": 3.0,
            "eval_zh_loss": 5.0,
            "train_loss": 3.0,
            "gradient_norm": 1.0,
        },
    ]
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return run


@pytest.mark.parametrize("phase", ["smoke", "pilot", "base"])
def test_a10_gate_accepts_matching_completed_run(tmp_path, gate, phase):
    module, _ = gate
    run = make_run(tmp_path, gate, phase)
    assert module.verify(run, phase=phase, root=PROJECT_ROOT)["status"] == "passed"


@pytest.mark.parametrize(
    ("file", "field", "value", "message"),
    [
        ("run_manifest.json", "status", "running", "not completed"),
        ("run_manifest.json", "config_sha256", "wrong", "identity mismatch"),
        ("training_result.json", "global_step", 1, "budget incomplete"),
        ("training_result.json", "tokens_seen", 1, "coverage"),
    ],
)
def test_a10_gate_rejects_identity_or_budget_drift(
    tmp_path, gate, file, field, value, message
):
    module, _ = gate
    run = make_run(tmp_path, gate)
    path = run / file
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError, match=message):
        module.verify(run, phase="smoke", root=PROJECT_ROOT)


def test_a10_gate_rejects_other_hardware(tmp_path, gate):
    module, _ = gate
    run = make_run(tmp_path, gate)
    path = run / "runtime_environment.json"
    data = json.loads(path.read_text())
    data["accelerator"]["devices"][0]["name"] = "NVIDIA GeForce RTX 4090"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError, match="single NVIDIA A10"):
        module.verify(run, phase="smoke", root=PROJECT_ROOT)


def test_formal_a10_gate_rejects_dirty_source(tmp_path, gate):
    module, _ = gate
    run = make_run(tmp_path, gate, "base")
    path = run / "runtime_environment.json"
    data = json.loads(path.read_text())
    data["code"]["dirty"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError, match="not clean"):
        module.verify(run, phase="base", root=PROJECT_ROOT)


def test_a10_gate_rejects_nonfinite_loss(tmp_path, gate):
    module, _ = gate
    run = make_run(tmp_path, gate)
    path = run / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["train_loss"] = float("nan")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ContractError, match="non-finite train_loss"):
        module.verify(run, phase="smoke", root=PROJECT_ROOT)


def test_resume_state_comparison_and_interruption(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "a10_resume_test", PROJECT_ROOT / "scripts/a10_resume_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = {"optimizer": [torch.tensor([1.0]), {"step": 4}]}
    second = {"optimizer": [torch.tensor([1.0]), {"step": 4}]}
    assert module.state_equal(first, second)
    second["optimizer"][0][0] = 2.0
    assert not module.state_equal(first, second)
    assert not module.state_equal([1], (1,))
    module.interrupt_after_fifth_update({"step": 4, "train_loss": 1.0})
    with pytest.raises(module.PlannedInterruption):
        module.interrupt_after_fifth_update({"step": 5, "train_loss": 1.0})
