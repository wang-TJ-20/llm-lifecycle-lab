from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import llm_lifecycle_lab.evaluation.native as native_evaluation
from llm_lifecycle_lab.contracts import Stage
from llm_lifecycle_lab.evaluation.native import NativeEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeTransformer:
    loaded_from: Path | None = None

    def __init__(self, config: object) -> None:
        self.config = config

    def load(self, path: Path) -> None:
        type(self).loaded_from = path

    def to_device(self, _device: object) -> None:
        return None

    def set_training(self, _training: bool) -> None:
        return None


def test_native_evaluator_loads_canonical_release_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provenance = {
        "package": {"model_route": "native", "stage": "pretrain"},
        "source_run": {"run_id": "native-60m-a10-base-v1-s42"},
        "checkpoint": {
            "checkpoint_id": "step-00045191",
            "global_step": 45191,
            "tokenizer_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "created_at": "2026-11-01T00:00:00+00:00",
        },
    }
    (tmp_path / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/models/tiny-60m.yaml").read_text(encoding="utf-8")
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.pt").touch()
    tokenizer = SimpleNamespace(
        manifest=SimpleNamespace(content_sha256="a" * 64),
        vocab_size=config["vocab_size"],
    )
    monkeypatch.setattr(
        native_evaluation.NativeTokenizer,
        "from_directory",
        lambda _path: tokenizer,
    )

    monkeypatch.setattr(
        native_evaluation,
        "NativeTransformer",
        _FakeTransformer,
    )
    monkeypatch.setattr(
        native_evaluation,
        "sha256_file",
        lambda _path: "f" * 64,
    )

    evaluator = NativeEvaluator(tmp_path, device="cpu")

    assert evaluator.metadata.stage is Stage.PRETRAIN
    assert evaluator.metadata.run_id == "native-60m-a10-base-v1-s42"
    assert evaluator.identity["weights_sha256"] == "f" * 64
    assert _FakeTransformer.loaded_from == tmp_path.resolve()
