"""Group-relative policy optimization with deterministic programmatic rewards."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import torch

from llm_lifecycle_lab.data.grpo import reward_response, rollout_seed
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.protocol import GenerationConfig, ModelProtocol
from llm_lifecycle_lab.training.objective import ObjectiveOutput


def masked_token_logps(
    model: ModelProtocol,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model.forward(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    )
    targets = labels[:, 1:]
    mask = targets != -100
    if bool((mask.sum(dim=-1) == 0).any()):
        raise ContractError("GRPO rollout has no response targets")
    safe = targets.masked_fill(~mask, 0)
    logps = (
        torch.log_softmax(output.logits[:, :-1].float(), dim=-1)
        .gather(-1, safe.unsqueeze(-1))
        .squeeze(-1)
    )
    return logps, mask


@dataclass(frozen=True, slots=True)
class GRPOSettings:
    group_size: int = 4
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    clip_epsilon: float = 0.2
    kl_beta: float = 0.04
    advantage_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ContractError("GRPO group_size must be at least 2")
        if self.max_new_tokens <= 0:
            raise ContractError("GRPO max_new_tokens must be positive")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ContractError("invalid GRPO sampling settings")
        if not 0 < self.clip_epsilon < 1:
            raise ContractError("GRPO clip_epsilon must be in (0, 1)")
        if self.kl_beta < 0 or self.advantage_epsilon <= 0:
            raise ContractError("invalid GRPO KL/advantage settings")


class GRPOObjective:
    name = "grpo"

    def __init__(
        self,
        *,
        tokenizer: Any,
        reference_model: ModelProtocol,
        settings: GRPOSettings,
    ) -> None:
        self.tokenizer = tokenizer
        self.reference = reference_model
        self.settings = settings
        self.last_rollouts: list[dict[str, Any]] = []
        for _, parameter in self.reference.trainable_parameters():
            parameter.requires_grad_(False)
        self.reference.set_training(False)

    def _rollouts(
        self, model: ModelProtocol, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        trainable = tuple(model.trainable_parameters())
        if not trainable:
            raise ContractError("GRPO policy has no trainable parameters")
        device = trainable[0][1].device
        sequences = []
        labels = []
        rewards = []
        records = []
        for row in range(len(batch["prompt_ids"])):
            prompt_ids = batch["prompt_ids"][row]
            for group_index in range(self.settings.group_size):
                output = model.generate(
                    input_ids=torch.tensor([prompt_ids], device=device),
                    attention_mask=None,
                    config=GenerationConfig(
                        max_new_tokens=self.settings.max_new_tokens,
                        do_sample=True,
                        temperature=self.settings.temperature,
                        top_p=self.settings.top_p,
                        seed=rollout_seed(batch["example_id"][row], group_index),
                        eos_token_id=self.tokenizer.chat_end_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    ),
                )
                sequence = output.token_ids[0].tolist()
                response = sequence[len(prompt_ids) :]
                content = (
                    response[:-1]
                    if response and response[-1] == self.tokenizer.chat_end_token_id
                    else response
                )
                text = self.tokenizer.decode(content, skip_special_tokens=False)
                reward = reward_response(
                    text, batch["answer"][row], batch["verifier"][row]
                )
                sequences.append(sequence)
                labels.append([-100] * len(prompt_ids) + response)
                rewards.append(reward)
                records.append(
                    {
                        "example_id": batch["example_id"][row],
                        "group_index": group_index,
                        "language": batch["language"][row],
                        "output": text,
                        "reward": reward,
                        "stop_reason": output.stop_reason,
                    }
                )
        length = max(map(len, sequences))
        padded_ids = torch.tensor(
            [
                sequence + [self.tokenizer.pad_token_id] * (length - len(sequence))
                for sequence in sequences
            ],
            device=device,
        )
        attention_mask = torch.tensor(
            [
                [True] * len(sequence) + [False] * (length - len(sequence))
                for sequence in sequences
            ],
            device=device,
        )
        padded_labels = torch.tensor(
            [label + [-100] * (length - len(label)) for label in labels],
            device=device,
        )
        return (
            padded_ids,
            padded_labels,
            attention_mask,
            records,
        )

    def __call__(self, model: ModelProtocol, batch: dict[str, Any]) -> ObjectiveOutput:
        was_training = model.is_training()
        model.set_training(False)
        try:
            input_ids, labels, attention_mask, records = self._rollouts(model, batch)
            rewards = torch.tensor(
                [record["reward"] for record in records],
                dtype=torch.float32,
                device=input_ids.device,
            )
            grouped = rewards.reshape(-1, self.settings.group_size)
            means = grouped.mean(dim=1, keepdim=True)
            deviations = grouped.std(dim=1, correction=0, keepdim=True)
            advantages = (
                (grouped - means) / (deviations + self.settings.advantage_epsilon)
            ).reshape(-1)
            with torch.no_grad():
                old_logps, mask = masked_token_logps(
                    model,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                self.reference.to_device(input_ids.device)
                reference_logps, reference_mask = masked_token_logps(
                    self.reference,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                if not torch.equal(mask, reference_mask):
                    raise ContractError("policy and reference GRPO masks differ")
        finally:
            model.set_training(was_training)
        current_logps, current_mask = masked_token_logps(
            model,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        if not torch.equal(mask, current_mask):
            raise ContractError("old and current policy GRPO masks differ")
        ratio = torch.exp((current_logps - old_logps).clamp(-20, 20))
        advantage = advantages[:, None]
        unclipped = ratio * advantage
        clipped = (
            ratio.clamp(1 - self.settings.clip_epsilon, 1 + self.settings.clip_epsilon)
            * advantage
        )
        reference_delta = (reference_logps - current_logps).clamp(-20, 20)
        kl = torch.exp(reference_delta) - reference_delta - 1
        objective = torch.minimum(unclipped, clipped) - self.settings.kl_beta * kl
        counts = current_mask.sum(dim=-1)
        per_sequence = (objective * current_mask).sum(dim=-1) / counts
        loss = -per_sequence.mean()
        self.last_rollouts = records
        zero_variance = float(
            (deviations.squeeze(-1) <= self.settings.advantage_epsilon).float().mean()
        )
        return ObjectiveOutput(
            loss=loss,
            metrics={
                "supervised_tokens": float(current_mask.sum()),
                "normalization_count": float(len(records)),
                "report_reward_mean": statistics.fmean(
                    record["reward"] for record in records
                ),
                "report_success_rate": float((rewards > 0).float().mean()),
                "report_zero_variance_groups": zero_variance,
                "report_approx_kl": float(
                    (kl.detach() * current_mask).sum() / current_mask.sum()
                ),
                "report_clip_fraction": float(
                    (
                        ((ratio.detach() - 1).abs() > self.settings.clip_epsilon)
                        * current_mask
                    ).sum()
                    / current_mask.sum()
                ),
            },
        )
