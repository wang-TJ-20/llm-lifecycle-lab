"""Bounded CLI conversation state for Native and HF backends."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.protocol import GenerationConfig


def generate_text(
    evaluator,
    ids: list[int],
    *,
    generation: GenerationConfig,
    chat: bool,
) -> dict[str, Any]:
    if len(ids) + generation.max_new_tokens > evaluator.config.max_sequence_length:
        raise ContractError("context is full; reduce the prompt/token budget")
    tokenizer = evaluator.tokenizer
    end = tokenizer.chat_end_token_id if chat else tokenizer.eos_token_id
    output = evaluator.model.generate(
        input_ids=torch.tensor([ids], device=evaluator.device),
        attention_mask=None,
        config=replace(
            generation,
            eos_token_id=end,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )
    generated = output.token_ids[0, len(ids) :].tolist()
    if generated and generated[-1] == end:
        generated = generated[:-1]
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=False),
        "prompt_tokens": len(ids),
        "generated_tokens": output.generated_tokens,
        "stop_reason": output.stop_reason,
    }


class ChatSession:
    def __init__(
        self,
        evaluator,
        *,
        mode: str = "auto",
        system: str | None = None,
        generation: GenerationConfig,
        history_policy: str = "error",
    ) -> None:
        self.evaluator = evaluator
        self.mode = (
            ("completion" if evaluator.identity["stage"] == "pretrain" else "chat")
            if mode == "auto"
            else mode
        )
        if self.mode not in {"completion", "chat"}:
            raise ContractError("mode must be auto, completion or chat")
        if system and self.mode != "chat":
            raise ContractError("system prompt requires chat mode")
        if history_policy not in {"error", "drop-oldest"}:
            raise ContractError("unknown history policy")
        self.system = system
        self.generation = generation
        self.history_policy = history_policy
        self.history: list[dict[str, str]] = []

    def reset(self) -> None:
        self.history = []

    def respond(self, prompt: str) -> dict[str, Any]:
        if not prompt.strip():
            raise ContractError("prompt cannot be empty")
        evaluator = self.evaluator
        tokenizer = evaluator.tokenizer
        history = list(self.history)
        dropped = 0
        while True:
            messages = (
                ([{"role": "system", "content": self.system}] if self.system else [])
                + history
                + [{"role": "user", "content": prompt}]
            )
            ids = (
                evaluator.prompt_ids(messages, "native-chat-v1")
                if self.mode == "chat"
                else tokenizer.encode(prompt, add_bos=True)
            )
            if len(ids) + self.generation.max_new_tokens <= (
                evaluator.config.max_sequence_length
            ):
                break
            if self.history_policy != "drop-oldest" or not history:
                raise ContractError(
                    "context is full; reset or reduce the prompt/token budget"
                )
            history = history[2:]
            dropped += 1
        generated = generate_text(
            evaluator,
            ids,
            generation=self.generation,
            chat=self.mode == "chat",
        )
        if self.mode == "chat":
            self.history = history + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": generated["text"]},
            ]
        return {
            "prompt": prompt,
            "mode": self.mode,
            "dropped_turns": dropped,
            **generated,
        }
