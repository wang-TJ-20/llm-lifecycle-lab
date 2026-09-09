"""Native decoder-only Transformer used by the teaching route."""

from llm_lifecycle_lab.model.native.config import (
    NativeModelConfig,
    load_native_model_config,
)
from llm_lifecycle_lab.model.native.transformer import NativeTransformer

__all__ = [
    "NativeModelConfig",
    "NativeTransformer",
    "load_native_model_config",
]
