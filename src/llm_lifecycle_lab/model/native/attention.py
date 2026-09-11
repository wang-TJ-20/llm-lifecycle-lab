"""结合 RoPE 位置编码的分组查询因果注意力。

Grouped-query causal attention with rotary position embeddings.
"""

from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.native.config import NativeModelConfig
from llm_lifecycle_lab.model.native.normalization import RMSNorm

KVCacheEntry: TypeAlias = tuple[Tensor, Tensor]


class RotaryEmbedding(nn.Module):
    """Model-level RoPE cache shared by every attention layer."""

    def __init__(
        self,
        head_dim: int,
        max_sequence_length: int,
        theta: float,
    ) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer(
            "cosine",
            embeddings.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sine",
            embeddings.sin(),
            persistent=False,
        )

    def forward(
        self,
        *,
        start_position: int,
        sequence_length: int,
    ) -> tuple[Tensor, Tensor]:
        stop_position = start_position + sequence_length
        if start_position < 0 or stop_position > self.cosine.shape[0]:
            raise ContractError("RoPE positions exceed the cached sequence length")
        return (
            self.cosine[start_position:stop_position][None, None, :, :],
            self.sine[start_position:stop_position][None, None, :, :],
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: NativeModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.query_groups
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = (
            RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        )
        self.k_norm = (
            RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        is_causal: bool,
        past_key_value: KVCacheEntry | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCacheEntry | None]:
        batch_size, sequence_length, _ = hidden_states.shape
        # 分头：[B, T, D] -> [B, Hq/Hkv, T, Dh]。Split query and KV heads.
        query = self._shape_query(self.q_proj(hidden_states))
        key = self._shape_key_value(self.k_proj(hidden_states))
        value = self._shape_key_value(self.v_proj(hidden_states))

        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)

        # RoPE 只旋转 Q/K，保持 V 不变。Rotate Q/K, not V.
        cosine, sine = position_embeddings
        cosine = cosine.to(dtype=query.dtype)
        sine = sine.to(dtype=query.dtype)
        query = apply_rotary_embedding(query, cosine, sine)
        key = apply_rotary_embedding(key, cosine, sine)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        # Cache 保留 Hkv 个头；仅计算时扩展到 Hq。Cache compact KV heads.
        present = (key, value) if use_cache else None

        repeated_key = repeat_key_value(key, self.num_key_value_groups)
        repeated_value = repeat_key_value(value, self.num_key_value_groups)
        attended = F.scaled_dot_product_attention(
            query,
            repeated_key,
            repeated_value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, sequence_length, -1)
        return self.o_proj(attended), present

    def _shape_query(self, tensor: Tensor) -> Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _shape_key_value(self, tensor: Tensor) -> Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)


def apply_rotary_embedding(
    tensor: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> Tensor:
    return (tensor * cosine) + (rotate_half(tensor) * sine)


def rotate_half(tensor: Tensor) -> Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def repeat_key_value(tensor: Tensor, repetitions: int) -> Tensor:
    if repetitions == 1:
        return tensor
    return tensor.repeat_interleave(repetitions, dim=1)


def build_attention_mask(
    attention_mask: Tensor | None,
    *,
    query_length: int,
    key_length: int,
    past_length: int,
    device: torch.device,
) -> Tensor | None:
    """构造一次供全部层共享的 [B, 1, Tq, Tk] 布尔 mask。

    Build one broadcastable boolean mask for all layers; True means allowed.
    """
    # CPU 基准选择共享显式 mask；加速设备使用 causal 内核。CPU keeps a shared mask.
    if attention_mask is None and past_length == 0 and device.type != "cpu":
        return None

    # Cache 续写的位置从 past_length 开始。Offset queries by the cached prefix.
    query_positions = past_length + torch.arange(query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    allowed = allowed[None, None, :, :]

    if attention_mask is None:
        return allowed
    if attention_mask.ndim != 2 or attention_mask.shape[1] != key_length:
        raise ContractError("attention_mask must have shape [batch, total_key_length]")
    return allowed & attention_mask[:, None, None, :].to(dtype=torch.bool)
