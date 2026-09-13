"""Direct Preference Optimization over frozen reference sequence log-probabilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.protocol import ModelProtocol
from llm_lifecycle_lab.training.objective import ObjectiveOutput


def response_logps(
    model: ModelProtocol,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model.forward(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    )
    targets = labels[:, 1:]
    mask = targets != -100
    counts = mask.sum(dim=-1)
    if bool((counts == 0).any()):
        raise ContractError("DPO sequence has zero response targets")
    safe_targets = targets.masked_fill(~mask, 0)
    token_logps = (
        F.log_softmax(output.logits[:, :-1].float(), dim=-1)
        .gather(-1, safe_targets.unsqueeze(-1))
        .squeeze(-1)
    )
    return (token_logps * mask).sum(dim=-1), counts


class DPOObjective:
    name = "dpo"

    def __init__(self, beta: float = 0.1) -> None:
        if not 0 < beta <= 1:
            raise ContractError("DPO beta must be in (0, 1]")
        self.beta = beta

    def __call__(self, model: ModelProtocol, batch: dict[str, Any]) -> ObjectiveOutput:
        required = {"reference_chosen_logps", "reference_rejected_logps"}
        if not required <= batch.keys():
            raise ContractError(
                "DPO batch is missing frozen reference log-probabilities"
            )
        chosen, chosen_tokens = response_logps(
            model,
            input_ids=batch["chosen_input_ids"],
            labels=batch["chosen_labels"],
            attention_mask=batch.get("chosen_attention_mask"),
        )
        rejected, rejected_tokens = response_logps(
            model,
            input_ids=batch["rejected_input_ids"],
            labels=batch["rejected_labels"],
            attention_mask=batch.get("rejected_attention_mask"),
        )
        reference_chosen = batch["reference_chosen_logps"].to(chosen)
        reference_rejected = batch["reference_rejected_logps"].to(rejected)
        policy_margin = chosen - rejected
        reference_margin = reference_chosen - reference_rejected
        reward_margin = self.beta * (policy_margin - reference_margin)
        loss = -F.logsigmoid(reward_margin).mean()
        pairs = chosen.shape[0]
        return ObjectiveOutput(
            loss=loss,
            metrics={
                "supervised_tokens": float(chosen_tokens.sum() + rejected_tokens.sum()),
                "normalization_count": float(pairs),
                "report_pair_accuracy": float(
                    (policy_margin.detach() > 0).float().mean()
                ),
                "report_reference_accuracy": float(
                    (reference_margin.detach() > 0).float().mean()
                ),
                "report_reward_margin": float(reward_margin.detach().mean()),
            },
        )
