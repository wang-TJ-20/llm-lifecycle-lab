"""Backend-local sampling helpers; generation policy stays caller-controlled."""

from __future__ import annotations

import torch
from torch import Tensor

from llm_lifecycle_lab.model.protocol import GenerationConfig


def make_generator(
    device: torch.device,
    seed: int,
) -> torch.Generator | None:
    try:
        generator = torch.Generator(device=device)
    except RuntimeError:
        torch.manual_seed(seed)
        return None
    generator.manual_seed(seed)
    return generator


def select_next_token(
    logits: Tensor,
    *,
    config: GenerationConfig,
    generator: torch.Generator | None,
) -> Tensor:
    if not config.do_sample:
        return logits.argmax(dim=-1)

    scaled = logits / config.temperature
    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            scaled,
            descending=True,
            dim=-1,
        )
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities > config.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        probabilities = torch.softmax(sorted_logits, dim=-1)
        sampled_index = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        )
        return sorted_indices.gather(-1, sampled_index).squeeze(-1)

    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    ).squeeze(-1)
