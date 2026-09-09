"""Model boundaries shared by native and adapted model implementations."""

from llm_lifecycle_lab.model.bundle import ModelBundle
from llm_lifecycle_lab.model.protocol import (
    GenerationConfig,
    GenerationOutput,
    ModelOutput,
    ModelProtocol,
)

__all__ = [
    "GenerationConfig",
    "GenerationOutput",
    "ModelBundle",
    "ModelOutput",
    "ModelProtocol",
]
