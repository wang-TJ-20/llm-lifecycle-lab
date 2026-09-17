from __future__ import annotations

import pytest
import torch

from llm_lifecycle_lab.data.grpo import reward_response, rollout_seed
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.native import NativeModelConfig, NativeTransformer
from llm_lifecycle_lab.training.stages import GRPOObjective, GRPOSettings


class TinyTokenizer:
    bos_token_id = 1
    pad_token_id = 0
    chat_end_token_id = 2

    @staticmethod
    def decode(ids, *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(97 + int(token) % 20) for token in ids)


def tiny_model() -> NativeTransformer:
    torch.manual_seed(9)
    return NativeTransformer(
        NativeModelConfig(
            model_id="tiny-grpo",
            vocab_size=32,
            num_hidden_layers=1,
            hidden_size=24,
            num_attention_heads=3,
            num_key_value_heads=1,
            intermediate_size=48,
            max_sequence_length=32,
        )
    )


@pytest.mark.parametrize(
    ("text", "answer", "verifier", "expected"),
    [
        (" yes ", "yes", "exact", 1.0),
        ("yes.", "yes", "exact", 0.0),
        ("+03", "3", "integer", 1.0),
        ("3 things", "3", "integer", 0.0),
        ('{"a":1}', '{"a":1}', "json", 1.0),
        ('{"a":1,"a":1}', '{"a":1}', "json", 0.0),
        ('{"a":NaN}', '{"a":1}', "json", 0.0),
        ("1", "1", "json", 0.0),
    ],
)
def test_programmatic_rewards(
    text: str, answer: str, verifier: str, expected: float
) -> None:
    assert reward_response(text, answer, verifier) == expected


def test_rollout_seed_is_stable_and_group_specific() -> None:
    assert rollout_seed("example", 0) == rollout_seed("example", 0)
    assert rollout_seed("example", 0) != rollout_seed("example", 1)
    with pytest.raises(ContractError, match="non-negative"):
        rollout_seed("example", -1)


def test_grpo_zero_variance_group_is_finite_and_reference_is_frozen() -> None:
    policy = tiny_model()
    reference = tiny_model()
    reference.load_state_dict(policy.state_dict())
    objective = GRPOObjective(
        tokenizer=TinyTokenizer(),
        reference_model=reference,
        settings=GRPOSettings(group_size=4, max_new_tokens=3),
    )
    output = objective(
        policy,
        {
            "example_id": ["one"],
            "prompt": ["prompt"],
            "prompt_ids": [[1, 7, 8]],
            "answer": ["unreachable answer"],
            "verifier": ["exact"],
            "language": ["en"],
        },
    )
    assert torch.isfinite(output.loss)
    assert output.metrics["normalization_count"] == 4
    assert output.metrics["supervised_tokens"] == 12
    assert output.metrics["report_reward_mean"] == 0
    assert output.metrics["report_zero_variance_groups"] == 1
    assert "report_clip_fraction" not in output.metrics
    assert all(not parameter.requires_grad for parameter in reference.parameters())


@pytest.mark.parametrize(
    "settings",
    [
        {"group_size": 1},
        {"max_new_tokens": 0},
        {"temperature": 0},
        {"top_p": 2},
        {"kl_beta": -1},
    ],
)
def test_grpo_rejects_invalid_settings(settings: dict) -> None:
    with pytest.raises(ContractError):
        GRPOSettings(**settings)
