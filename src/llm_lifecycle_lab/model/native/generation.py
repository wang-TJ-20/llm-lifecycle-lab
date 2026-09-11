"""使用 KV Cache 逐 token 生成，并按调用方设置进行采样。

Generate tokens with a KV cache and caller-controlled sampling settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.protocol import GenerationConfig, GenerationOutput

if TYPE_CHECKING:
    from llm_lifecycle_lab.model.native.transformer import NativeTransformer


@torch.inference_mode()
def generate_tokens(
    model: NativeTransformer,
    *,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    config: GenerationConfig,
) -> GenerationOutput:
    """仅支持无 padding 或左侧 padding 的非空 prompt。

    Accept nonempty prompts with no padding or left padding only.
    """
    if input_ids.ndim != 2 or 0 in input_ids.shape:
        raise ContractError(
            "generation input_ids must have nonempty shape [batch, time]"
        )
    if input_ids.shape[1] + config.max_new_tokens > model.config.max_sequence_length:
        raise ContractError(
            "prompt plus max_new_tokens exceeds model max_sequence_length"
        )
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ContractError("generation attention_mask must match input_ids shape")
        if attention_mask.device != input_ids.device:
            raise ContractError(
                "attention_mask and input_ids must be on the same device"
            )
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ContractError("attention_mask must contain only 0 or 1")
        attention_mask = attention_mask.to(dtype=torch.bool)
        if not bool(attention_mask[:, -1].all()) or bool(
            (attention_mask[:, :-1] & ~attention_mask[:, 1:]).any()
        ):
            raise ContractError(
                "generation requires nonempty prompts with left padding only"
            )
        if bool(attention_mask.all()):
            attention_mask = None
    for name in ("eos_token_id", "pad_token_id"):
        token_id = getattr(config, name)
        if token_id is not None and (
            type(token_id) is not int or not 0 <= token_id < model.config.vocab_size
        ):
            raise ContractError(f"{name} must be an integer inside the vocabulary")

    generated = input_ids
    finished = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )
    cache = None
    next_input = input_ids
    generated_steps = 0
    stop_reason = "length"
    pad_token_id = config.pad_token_id
    if pad_token_id is None:
        pad_token_id = config.eos_token_id if config.eos_token_id is not None else 0

    was_training = model.training
    model.eval()
    try:
        generator = (
            make_generator(input_ids.device, config.seed) if config.do_sample else None
        )
        for _ in range(config.max_new_tokens):
            # 首轮处理整个 prompt，随后只输入新 token。Prefill, then one-token decode.
            output = model(
                input_ids=next_input,
                attention_mask=attention_mask,
                use_cache=True,
                cache=cache,
                logits_to_keep=1,
            )
            next_token = select_next_token(
                output.logits[:, -1, :], config=config, generator=generator
            )
            next_token = torch.where(
                finished, torch.full_like(next_token, pad_token_id), next_token
            )
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (input_ids.shape[0], 1),
                            dtype=torch.bool,
                            device=input_ids.device,
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
        model.train(was_training)

    return GenerationOutput(
        token_ids=generated,
        prompt_tokens=input_ids.shape[1],
        generated_tokens=generated_steps,
        stop_reason=stop_reason,
        metadata={"batch_size": input_ids.shape[0]},
    )


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
