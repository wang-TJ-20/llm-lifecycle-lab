"""Decoder-only native language model shared by the 10M and 60M profiles."""

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
from llm_lifecycle_lab.model.native.attention import KVCacheEntry, RotaryEmbedding
from llm_lifecycle_lab.model.native.config import NativeModelConfig
from llm_lifecycle_lab.model.native.generation import (
    make_generator,
    select_next_token,
)
from llm_lifecycle_lab.model.native.layers import TransformerBlock
from llm_lifecycle_lab.model.native.normalization import RMSNorm
from llm_lifecycle_lab.model.protocol import (
    GenerationConfig,
    GenerationOutput,
    ModelOutput,
)

KVCache = tuple[KVCacheEntry, ...]


class NativeTransformer(nn.Module):
    """A compact Llama-style decoder with no stage-specific loss logic."""

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
    ) -> ModelOutput:
        self._validate_inputs(input_ids, attention_mask, cache)
        hidden_states = self.token_embedding(input_ids)
        present_cache: list[KVCacheEntry] = []
        past_length = 0 if cache is None else cache[0][0].shape[2]
        position_embeddings = self.rotary(
            start_position=past_length,
            sequence_length=input_ids.shape[1],
        )

        for index, layer in enumerate(self.layers):
            past_key_value = None if cache is None else cache[index]
            hidden_states, present = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if present is not None:
                present_cache.append(present)

        logits = self.lm_head(self.final_norm(hidden_states))
        return ModelOutput(
            logits=logits,
            cache=tuple(present_cache) if use_cache else None,
        )

    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        config: GenerationConfig,
    ) -> GenerationOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ContractError("generation input_ids must have shape [batch, time]")
        if input_ids.shape[1] + config.max_new_tokens > self.config.max_sequence_length:
            raise ContractError(
                "prompt plus max_new_tokens exceeds model max_sequence_length"
            )

        was_training = self.training
        self.eval()
        generated = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        generator = make_generator(input_ids.device, config.seed)
        finished = torch.zeros(
            input_ids.shape[0],
            dtype=torch.bool,
            device=input_ids.device,
        )
        cache: KVCache | None = None
        next_input = input_ids
        generated_steps = 0
        stop_reason = "length"
        pad_token_id = (
            config.pad_token_id
            if config.pad_token_id is not None
            else config.eos_token_id
        )
        if pad_token_id is None:
            pad_token_id = 0

        try:
            for _ in range(config.max_new_tokens):
                output = self.forward(
                    input_ids=next_input,
                    attention_mask=attention_mask,
                    use_cache=True,
                    cache=cache,
                )
                next_token = select_next_token(
                    output.logits[:, -1, :],
                    config=config,
                    generator=generator,
                )
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, pad_token_id),
                    next_token,
                )
                generated = torch.cat((generated, next_token[:, None]), dim=1)
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=torch.bool,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                cache = output.cache
                next_input = next_token[:, None]
                generated_steps += 1

                if config.eos_token_id is not None:
                    finished |= next_token == config.eos_token_id
                    if bool(finished.all()):
                        stop_reason = "eos"
                        break
        finally:
            self.train(was_training)

        return GenerationOutput(
            token_ids=generated,
            prompt_tokens=input_ids.shape[1],
            generated_tokens=generated_steps,
            stop_reason=stop_reason,
            metadata={"batch_size": input_ids.shape[0]},
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
