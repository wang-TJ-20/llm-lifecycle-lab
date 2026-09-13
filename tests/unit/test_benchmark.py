from __future__ import annotations

import pytest
import torch

from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.interop.benchmark import (
    generation_inputs,
    measure_model,
    select_quantized_engine,
)


class FakeTokenizer:
    def __call__(self, text: str, *, return_tensors: str):
        assert text
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "token_type_ids": torch.zeros((1, 2), dtype=torch.long),
        }


def test_generation_inputs_filters_unsupported_tokenizer_fields() -> None:
    inputs = generation_inputs(FakeTokenizer(), "hello")
    assert set(inputs) == {"input_ids", "attention_mask"}


def test_generation_inputs_requires_input_ids() -> None:
    with pytest.raises(ContractError, match="input_ids"):
        generation_inputs(lambda *_args, **_kwargs: {}, "hello")


@pytest.mark.parametrize(
    ("max_new_tokens", "repeats"),
    [(0, 1), (1, 0)],
)
def test_measure_model_rejects_non_positive_protocol(
    max_new_tokens: int, repeats: int
) -> None:
    with pytest.raises(ContractError, match="must be positive"):
        measure_model(
            object(),
            object(),
            max_new_tokens=max_new_tokens,
            repeats=repeats,
        )


def test_quantized_engine_is_selected_or_rejected_explicitly() -> None:
    supported = set(torch.backends.quantized.supported_engines)
    usable = supported & {"x86", "fbgemm", "onednn", "qnnpack"}
    if usable:
        assert select_quantized_engine() in usable
    else:
        with pytest.raises(ContractError, match="no supported quantized CPU engine"):
            select_quantized_engine()
