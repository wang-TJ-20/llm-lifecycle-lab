from __future__ import annotations

from pathlib import Path

import pytest

import llm_lifecycle_lab.evaluation.native as native_evaluation
from llm_lifecycle_lab.contracts import Stage
from llm_lifecycle_lab.evaluation.native import NativeEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_PACKAGE = PROJECT_ROOT / "models/base-v1-final"


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
) -> None:
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

    evaluator = NativeEvaluator(BASE_PACKAGE, device="cpu")

    assert evaluator.metadata.stage is Stage.PRETRAIN
    assert evaluator.metadata.run_id == "native-60m-base-v1-s42"
    assert evaluator.identity["weights_sha256"] == "f" * 64
    assert _FakeTransformer.loaded_from == BASE_PACKAGE.resolve()
