"""Lifecycle objectives executed by the shared engine."""

from llm_lifecycle_lab.training.stages.pretrain import (
    PretrainObjective,
    causal_lm_loss,
)

__all__ = ["PretrainObjective", "causal_lm_loss"]
