"""组合归一化、注意力、SwiGLU 和残差连接，构建单层 Transformer。

Build one Transformer block with normalization, attention, SwiGLU, and residuals.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from llm_lifecycle_lab.model.native.attention import (
    CausalSelfAttention,
    KVCacheEntry,
)
from llm_lifecycle_lab.model.native.config import NativeModelConfig
from llm_lifecycle_lab.model.native.normalization import RMSNorm


class SwiGLU(nn.Module):
    def __init__(self, config: NativeModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        # 门控后投影回 D 维：[B, T, I] -> [B, T, D]。Gate, then project down.
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: NativeModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.post_attention_norm = RMSNorm(
            config.hidden_size,
            config.norm_eps,
        )
        self.mlp = SwiGLU(config)

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
        # Pre-norm 残差路径不改变 [B, T, D]。Both residuals preserve the shape.
        residual = hidden_states
        attention_output, present = self.attention(
            self.input_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            is_causal=is_causal,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attention_output
        hidden_states = hidden_states + self.mlp(
            self.post_attention_norm(hidden_states)
        )
        return hidden_states, present
