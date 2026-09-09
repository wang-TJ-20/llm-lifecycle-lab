"""Narrow model interface consumed by the shared training engine."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from llm_lifecycle_lab.exceptions import ContractError


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Backend-neutral output from a causal language model."""

    logits: Any
    cache: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Generation settings resolved and recorded by the caller."""

    max_new_tokens: int
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 42
    eos_token_id: int | None = None
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ContractError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ContractError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ContractError("top_p must be in (0, 1]")
        if self.seed < 0:
            raise ContractError("generation seed must be non-negative")


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    """Backend-neutral token output with recorded generation facts."""

    token_ids: Any
    prompt_tokens: int
    generated_tokens: int
    stop_reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.generated_tokens < 0:
            raise ContractError("generation token counts must be non-negative")
        if not self.stop_reason:
            raise ContractError("generation stop_reason must not be empty")


@runtime_checkable
class ModelProtocol(Protocol):
    """Only tensor-level model behavior and checkpoint state belong here."""

    def forward(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None = None,
        use_cache: bool = False,
        cache: Any | None = None,
    ) -> ModelOutput:
        """Run a forward pass without computing a stage-specific loss."""

    def generate(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None,
        config: GenerationConfig,
    ) -> GenerationOutput:
        """Generate token IDs from already-tokenized inputs."""

    def trainable_parameters(self) -> Iterable[tuple[str, Any]]:
        """Return named parameters that currently require gradients."""

    def set_training(self, training: bool) -> None:
        """Set train or evaluation behavior without exposing the backend."""

    def is_training(self) -> bool:
        """Return whether training-specific model behavior is enabled."""

    def to_device(self, device: Any) -> None:
        """Move model state to the execution device."""

    def save(self, path: Path) -> None:
        """Persist model state for a training checkpoint."""

    def load(self, path: Path) -> None:
        """Restore model state from a training checkpoint."""
