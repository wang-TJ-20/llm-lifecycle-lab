"""构建 Native Decoder-only 模型：Embedding、Transformer Blocks 和 LM Head。

Build the Native decoder from embeddings, Transformer blocks, and an LM head.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError, ContractError
from llm_lifecycle_lab.model.native.attention import (
    KVCacheEntry,
    RotaryEmbedding,
    build_attention_mask,
)
from llm_lifecycle_lab.model.native.config import NativeModelConfig
from llm_lifecycle_lab.model.native.generation import generate_tokens
from llm_lifecycle_lab.model.native.layers import TransformerBlock
from llm_lifecycle_lab.model.native.normalization import RMSNorm
from llm_lifecycle_lab.model.protocol import (
    GenerationConfig,
    GenerationOutput,
    ModelOutput,
)

KVCache = tuple[KVCacheEntry, ...]


class NativeTransformer(nn.Module):
    """模型只输出 logits，loss 在训练代码中计算。

    Return logits; the training code computes the task-specific loss.
    """

    def __init__(self, config: NativeModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.rotary = RotaryEmbedding(
            config.head_dim,
            config.max_sequence_length,
            config.rope_theta,
        )
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,
        cache: KVCache | None = None,
        logits_to_keep: int = 0,
    ) -> ModelOutput:
        """默认返回全部位置；logits_to_keep > 0 只投影末尾位置。

        Zero keeps all logits; a positive value projects only the final positions.
        KV cache always retains the complete input, regardless of this setting.
        """
        self._validate_inputs(input_ids, attention_mask, cache)
        if (
            type(logits_to_keep) is not int
            or not 0 <= logits_to_keep <= input_ids.shape[1]
        ):
            raise ContractError(
                "logits_to_keep must be an integer in [0, input length]"
            )
        # token ID 转为向量：[B, T] -> [B, T, D]。Token IDs to hidden states.
        hidden_states = self.token_embedding(input_ids)
        present_cache: list[KVCacheEntry] = []
        past_length = 0 if cache is None else cache[0][0].shape[2]
        position_embeddings = self.rotary(
            start_position=past_length,
            sequence_length=input_ids.shape[1],
        )
        # 各层共享 mask；加速设备可用 SDPA causal。Share masks across layers.
        mask = build_attention_mask(
            attention_mask,
            query_length=input_ids.shape[1],
            key_length=past_length + input_ids.shape[1],
            past_length=past_length,
            device=input_ids.device,
        )
        is_causal = mask is None and past_length == 0

        for index, layer in enumerate(self.layers):
            past_key_value = None if cache is None else cache[index]
            hidden_states, present = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                is_causal=is_causal,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if present is not None:
                present_cache.append(present)

        # 先切位置再做词表投影：[B, T/1, D] -> [B, T/1, V]。Slice before LM head.
        if logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        logits = self.lm_head(self.final_norm(hidden_states))
        return ModelOutput(
            logits=logits,
            cache=tuple(present_cache) if use_cache else None,
        )

    def generate(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        config: GenerationConfig,
    ) -> GenerationOutput:
        return generate_tokens(
            self, input_ids=input_ids, attention_mask=attention_mask, config=config
        )

    def trainable_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

    def set_training(self, training: bool) -> None:
        self.train(training)

    def is_training(self) -> bool:
        return self.training

    def to_device(self, device: Any) -> None:
        self.to(device)

    def save(self, path: Path) -> None:
        target = Path(path)
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise ArtifactError(f"model checkpoint directory is not empty: {target}")
        target.mkdir(parents=True, exist_ok=True)
        _atomic_json(target / "config.json", asdict(self.config))
        _atomic_torch_save(target / "model.pt", self.state_dict())

    def load(self, path: Path) -> None:
        source = Path(path)
        try:
            saved_config = NativeModelConfig.from_dict(
                json.loads((source / "config.json").read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ConfigError) as exc:
            raise ArtifactError(
                f"cannot read model config from {source}: {exc}"
            ) from exc
        if saved_config != self.config:
            raise ArtifactError("checkpoint model config does not match this model")
        try:
            state = torch.load(
                source / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.load_state_dict(state, strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ArtifactError(
                f"cannot load model checkpoint {source}: {exc}"
            ) from exc

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

    def _validate_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        cache: KVCache | None,
    ) -> None:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ContractError("input_ids must have shape [batch, time]")
        if input_ids.dtype not in {torch.int32, torch.int64}:
            raise ContractError("input_ids must contain integer token IDs")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ContractError("input_ids contain values outside the vocabulary")
        if cache is not None and len(cache) != len(self.layers):
            raise ContractError("cache layer count does not match model")

        past_length = 0 if cache is None else cache[0][0].shape[2]
        total_length = past_length + input_ids.shape[1]
        if total_length > self.config.max_sequence_length:
            raise ContractError("input sequence exceeds max_sequence_length")
        if attention_mask is not None and attention_mask.shape != (
            input_ids.shape[0],
            total_length,
        ):
            raise ContractError(
                "attention_mask must have shape [batch, total_sequence_length]"
            )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    content = json.dumps(value, indent=2, sort_keys=True)
    _atomic_bytes(path, f"{content}\n".encode())


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(value, temporary)
        os.replace(temporary, path)
    except (OSError, RuntimeError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ArtifactError(f"cannot save model state to {path}: {exc}") from exc


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ArtifactError(f"cannot write {path}: {exc}") from exc
