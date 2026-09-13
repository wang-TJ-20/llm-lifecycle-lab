from __future__ import annotations

import math

import pytest
import torch

from llm_lifecycle_lab.data.dpo import collate_dpo, encode_dpo_pair
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.native import NativeModelConfig, NativeTransformer
from llm_lifecycle_lab.training.stages import DPOObjective, response_logps


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    chat_start_token_id = 3
    chat_end_token_id = 4

    @staticmethod
    def encode(text: str) -> list[int]:
        return [5 + ord(character) % 19 for character in text]

    def encode_chat(
        self, messages, *, add_generation_prompt: bool = False
    ) -> list[int]:
        ids = [self.bos_token_id]
        for message in messages:
            ids.extend([self.chat_start_token_id])
            ids.extend(self.encode(message["role"] + "\n"))
            ids.extend(self.encode(message["content"]))
            ids.extend([self.chat_end_token_id])
        if add_generation_prompt:
            ids.extend([self.chat_start_token_id])
            ids.extend(self.encode("assistant\n"))
        return ids


def tiny_model() -> NativeTransformer:
    torch.manual_seed(7)
    return NativeTransformer(
        NativeModelConfig(
            model_id="tiny-dpo",
            vocab_size=32,
            num_hidden_layers=1,
            hidden_size=24,
            num_attention_heads=3,
            num_key_value_heads=1,
            intermediate_size=48,
            max_sequence_length=128,
        )
    )


def test_dpo_equal_policy_and_reference_is_log_two() -> None:
    tokenizer = TinyTokenizer()
    examples = [
        encode_dpo_pair(
            "question",
            "chosen",
            "wrong",
            tokenizer=tokenizer,
            sequence_length=128,
            language="en",
            pair_id="one",
        ),
        encode_dpo_pair(
            "问题",
            "正确",
            "错误",
            tokenizer=tokenizer,
            sequence_length=128,
            language="zh",
            pair_id="two",
        ),
    ]
    batch = collate_dpo(examples, pad_token_id=tokenizer.pad_token_id)
    model = tiny_model()
    for choice in ("chosen", "rejected"):
        logps, _ = response_logps(
            model,
            input_ids=batch[f"{choice}_input_ids"],
            labels=batch[f"{choice}_labels"],
            attention_mask=batch[f"{choice}_attention_mask"],
        )
        batch[f"reference_{choice}_logps"] = logps.detach()
    output = DPOObjective(beta=0.1)(model, batch)
    assert float(output.loss.detach()) == pytest.approx(math.log(2), abs=1e-6)
    assert output.metrics["normalization_count"] == 2
    assert output.metrics["supervised_tokens"] > 2
    assert output.metrics["report_reward_margin"] == pytest.approx(0.0, abs=1e-7)


def test_dpo_masks_prompts_and_rejects_invalid_pairs() -> None:
    tokenizer = TinyTokenizer()
    pair = encode_dpo_pair(
        "prompt",
        "yes",
        "no",
        tokenizer=tokenizer,
        sequence_length=128,
        language="en",
        pair_id="one",
    )
    for choice in ("chosen", "rejected"):
        labels = pair[f"{choice}_labels"]
        first = int(torch.nonzero(labels != -100)[0])
        assert (labels[:first] == -100).all()
        assert int(labels[-1]) == tokenizer.chat_end_token_id
    with pytest.raises(ContractError, match="must differ"):
        encode_dpo_pair(
            "prompt",
            "same",
            " same ",
            tokenizer=tokenizer,
            sequence_length=128,
            language="en",
            pair_id="bad",
        )
    with pytest.raises(ContractError, match="beta"):
        DPOObjective(beta=0)
    with pytest.raises(ContractError, match="missing frozen reference"):
        DPOObjective()(tiny_model(), collate_dpo([pair], pad_token_id=0))
