"""校验 Native 教学模型的结构维度和数值参数。

Validate architecture dimensions and numeric settings for the Native Transformer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.config import load_mapping
from llm_lifecycle_lab.exceptions import ConfigError

_MODEL_FIELDS = {
    "model_id",
    "vocab_size",
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_sequence_length",
    "norm_eps",
    "rope_theta",
    "qk_norm",
    "attention_dropout",
    "tie_word_embeddings",
    "initializer_range",
}
_METADATA_FIELDS = {"schema_version", "model_route", "provider", "architecture"}


@dataclass(frozen=True, slots=True)
class NativeModelConfig:
    model_id: str
    vocab_size: int
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_sequence_length: int
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    qk_norm: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_sequence_length": self.max_sequence_length,
        }
        for name, value in integer_fields.items():
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ConfigError("model_id must be a non-empty string")
        for name in ("qk_norm", "tie_word_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a boolean")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ConfigError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ConfigError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2 != 0:
            raise ConfigError("attention head dimension must be even for RoPE")
        for name in ("norm_eps", "rope_theta", "initializer_range"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigError(f"{name} must be finite and positive")
        if type(self.attention_dropout) not in (int, float):
            raise ConfigError("attention_dropout must be a number in [0, 1)")
        if not 0 <= self.attention_dropout < 1:
            raise ConfigError("attention_dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def query_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def estimated_parameter_count(self) -> int:
        token_parameters = self.token_parameter_count_for_vocab_size(self.vocab_size)
        attention = (
            2 * self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * self.num_key_value_heads * self.head_dim
        )
        feed_forward = 3 * self.hidden_size * self.intermediate_size
        norms = 2 * self.hidden_size
        qk_norms = 2 * self.head_dim if self.qk_norm else 0
        return (
            token_parameters
            + self.num_hidden_layers * (attention + feed_forward + norms + qk_norms)
            + self.hidden_size
        )

    def token_parameter_count_for_vocab_size(self, vocab_size: int) -> int:
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ConfigError("vocab_size must be a positive integer")
        multiplier = 1 if self.tie_word_embeddings else 2
        return vocab_size * self.hidden_size * multiplier

    def parameter_count_for_vocab_size(self, vocab_size: int) -> int:
        current_token_parameters = self.token_parameter_count_for_vocab_size(
            self.vocab_size
        )
        return (
            self.estimated_parameter_count
            - current_token_parameters
            + self.token_parameter_count_for_vocab_size(vocab_size)
        )

    def token_parameter_share_for_vocab_size(self, vocab_size: int) -> float:
        return self.token_parameter_count_for_vocab_size(
            vocab_size
        ) / self.parameter_count_for_vocab_size(vocab_size)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NativeModelConfig:
        unknown = sorted(set(data) - _MODEL_FIELDS - _METADATA_FIELDS)
        if unknown:
            raise ConfigError(
                f"unknown native model config fields: {', '.join(unknown)}"
            )
        if data.get("provider", "native") != "native":
            raise ConfigError("native model config requires provider=native")
        if data.get("model_route", "native") != "native":
            raise ConfigError("native model config requires model_route=native")
        try:
            return cls(
                model_id=data["model_id"],
                vocab_size=data["vocab_size"],
                num_hidden_layers=data["num_hidden_layers"],
                hidden_size=data["hidden_size"],
                num_attention_heads=data["num_attention_heads"],
                num_key_value_heads=data["num_key_value_heads"],
                intermediate_size=data["intermediate_size"],
                max_sequence_length=data["max_sequence_length"],
                norm_eps=data.get("norm_eps", 1e-6),
                rope_theta=data.get("rope_theta", 10_000.0),
                qk_norm=data.get("qk_norm", False),
                attention_dropout=data.get("attention_dropout", 0.0),
                tie_word_embeddings=data.get("tie_word_embeddings", True),
                initializer_range=data.get("initializer_range", 0.02),
            )
        except KeyError as exc:
            raise ConfigError(
                f"missing native model config field: {exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid native model config value: {exc}") from exc


def load_native_model_config(path: str | Path) -> NativeModelConfig:
    return NativeModelConfig.from_dict(load_mapping(path))
