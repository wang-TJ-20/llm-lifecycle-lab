"""Causal language-model objective for native pretraining."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from llm_lifecycle_lab.data.packing import LANGUAGE_IDS
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.protocol import ModelProtocol
from llm_lifecycle_lab.training.logprobs import causal_token_log_probs
from llm_lifecycle_lab.training.objective import ObjectiveOutput


class PretrainObjective:
    name = "pretrain"

    def __call__(
        self,
        model: ModelProtocol,
        batch: dict[str, Any],
    ) -> ObjectiveOutput:
        labels = batch["labels"]
        output = model.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            use_cache=False,
        )
        token_log_probs, supervision_mask = causal_token_log_probs(
            output.logits,
            labels,
        )
        loss, supervised_tokens = _loss_from_token_log_probs(
            token_log_probs,
            supervision_mask,
        )
        source_bytes_value = batch.get("source_bytes")
        source_bytes = (
            float(source_bytes_value.sum())
            if isinstance(source_bytes_value, Tensor)
            else 0.0
        )
        metrics = {
            "loss": float(loss.detach()),
            "supervised_tokens": float(supervised_tokens),
            "source_bytes": float(source_bytes),
        }
        if source_bytes > 0:
            metrics["bits_per_byte"] = (
                float(loss.detach()) * supervised_tokens / math.log(2) / source_bytes
            )
        if not model.is_training():
            metrics.update(
                _language_metrics(
                    token_log_probs,
                    supervision_mask,
                    batch.get("language_ids"),
                    batch.get("byte_weights"),
                )
            )
        return ObjectiveOutput(
            loss=loss,
            metrics=metrics,
        )


def causal_lm_loss(
    logits: Tensor,
    labels: Tensor,
) -> tuple[Tensor, int]:
    token_log_probs, mask = causal_token_log_probs(logits, labels)
    return _loss_from_token_log_probs(token_log_probs, mask)


def _loss_from_token_log_probs(
    token_log_probs: Tensor,
    mask: Tensor,
) -> tuple[Tensor, int]:
    supervised_tokens = int(mask.sum())
    if supervised_tokens == 0:
        raise ContractError("pretrain batch has zero supervised tokens")
    loss = -token_log_probs.sum() / supervised_tokens
    if not bool(torch.isfinite(loss)):
        raise ContractError("pretrain loss is not finite")
    return loss, supervised_tokens


def _language_metrics(
    token_log_probs: Tensor,
    supervision_mask: Tensor,
    language_ids: Any,
    byte_weights: Any,
) -> dict[str, float]:
    if not isinstance(language_ids, Tensor) or not isinstance(
        byte_weights,
        Tensor,
    ):
        return {}
    shifted_languages = language_ids[:, 1:]
    shifted_byte_weights = byte_weights[:, 1:]
    if shifted_languages.shape != token_log_probs.shape:
        raise ContractError("pretrain language IDs do not align with labels")
    if shifted_byte_weights.shape != token_log_probs.shape:
        raise ContractError("pretrain byte weights do not align with labels")

    metrics: dict[str, float] = {}
    for language, language_id in LANGUAGE_IDS.items():
        mask = supervision_mask & (shifted_languages == language_id)
        token_count = int(mask.sum())
        if token_count == 0:
            continue
        metrics[f"language_{language}_nll_sum"] = float(
            -token_log_probs.masked_select(mask).sum()
        )
        metrics[f"language_{language}_tokens"] = float(token_count)
        metrics[f"language_{language}_source_bytes"] = float(
            shifted_byte_weights.masked_select(mask).sum()
        )
    return metrics
