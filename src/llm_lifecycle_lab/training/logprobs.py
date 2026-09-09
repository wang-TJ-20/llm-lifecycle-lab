"""Canonical causal token log-probability computation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def causal_token_log_probs(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[Tensor, Tensor]:
    """Return shifted token log probabilities and their supervision mask."""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [batch, time, vocab] and labels [batch, time]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels sequence dimensions must match")
    if logits.shape[1] < 2:
        raise ValueError("causal log probabilities require at least two tokens")

    shifted_logits = logits[:, :-1, :].to(dtype=torch.float32)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probabilities = F.log_softmax(shifted_logits, dim=-1)
    selected = log_probabilities.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    return selected.masked_fill(~mask, 0.0), mask
