"""Backend-neutral objective contract consumed by the shared engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor

from llm_lifecycle_lab.model.protocol import ModelProtocol


@dataclass(frozen=True, slots=True)
class ObjectiveOutput:
    loss: Tensor
    metrics: dict[str, float]


class TrainingObjective(Protocol):
    name: str

    def __call__(
        self,
        model: ModelProtocol,
        batch: dict[str, Any],
    ) -> ObjectiveOutput: ...
