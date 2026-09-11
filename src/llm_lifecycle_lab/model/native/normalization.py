"""使用 float32 均方计算的 RMSNorm，输出恢复为输入 dtype。

Compute RMSNorm statistics in float32 and restore the input dtype.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.to(dtype=torch.float32)
        # 最后一维取均方，不减均值。Mean square, not centered variance.
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return (self.weight.to(dtype=torch.float32) * normalized).to(dtype=input_dtype)
