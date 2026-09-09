"""Validated binding between a model and its route-specific assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_lifecycle_lab.contracts import ModelMetadata, ModelRoute
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.protocol import ModelProtocol


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """Bind model, tokenizer, template, and metadata for one run.

    The binding is immutable. The model object and its trainable parameters are
    intentionally mutable during training.
    """

    model: ModelProtocol
    tokenizer: Any
    chat_template: str | None
    metadata: ModelMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelProtocol):
            raise ContractError("model does not satisfy ModelProtocol")
        if self.tokenizer is None:
            raise ContractError("ModelBundle requires a tokenizer")
        for method_name in ("encode", "decode"):
            if not callable(getattr(self.tokenizer, method_name, None)):
                raise ContractError(
                    f"tokenizer must provide a callable {method_name} method"
                )

        if self.chat_template is not None and not self.chat_template.strip():
            raise ContractError("chat_template must be non-empty when provided")

        if self.metadata.model_route is ModelRoute.QWEN3_TRANSFER:
            self._validate_qwen_assets()

    def _validate_qwen_assets(self) -> None:
        if self.metadata.model_id != "Qwen/Qwen3-0.6B-Base":
            raise ContractError("qwen3-transfer requires model_id=Qwen/Qwen3-0.6B-Base")
        if self.metadata.upstream_revision != self.metadata.tokenizer_revision:
            raise ContractError("Qwen model and tokenizer revisions must match")
        if self.chat_template is None:
            raise ContractError("qwen3-transfer requires an explicit chat template")
        if not self.metadata.chat_template_version:
            raise ContractError(
                "qwen3-transfer requires chat_template_version metadata"
            )
