from __future__ import annotations

import pytest

from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.training.batching import (
    DeterministicTokenQuotaBatchStream,
    stable_stratified_subset,
)


def test_stable_stratified_subset_is_balanced_and_order_independent() -> None:
    examples = [
        {
            "id": f"{language}-{index}",
            "language": language,
        }
        for language in ("en", "zh")
        for index in range(8)
    ]
    selected = stable_stratified_subset(
        examples,
        limit=6,
        strata=("en", "zh"),
        stratum_field="language",
        identity_field="id",
    )
    reversed_selected = stable_stratified_subset(
        list(reversed(examples)),
        limit=6,
        strata=("en", "zh"),
        stratum_field="language",
        identity_field="id",
    )
    assert [item["id"] for item in selected] == [
        item["id"] for item in reversed_selected
    ]
    assert [item["language"] for item in selected] == [
        "en",
        "zh",
        "en",
        "zh",
        "en",
        "zh",
    ]


def test_stable_stratified_subset_requires_every_stratum() -> None:
    with pytest.raises(ContractError, match="every requested stratum"):
        stable_stratified_subset(
            [{"id": "en-1", "language": "en"}],
            limit=2,
            strata=("en", "zh"),
            stratum_field="language",
            identity_field="id",
        )


def test_token_quota_stream_tracks_tokens_and_restores_exactly() -> None:
    dataset = [
        {"id": "general", "task": "general", "tokens": 20},
        {"id": "short-a", "task": "short", "tokens": 2},
        {"id": "short-b", "task": "short", "tokens": 2},
    ]
    kwargs = {
        "batch_size": 3,
        "seed": 42,
        "collate_fn": lambda rows: [row["id"] for row in rows],
        "stratum_field": "task",
        "token_count_field": "tokens",
        "weights": {"general": 0.5, "short": 0.5},
    }
    stream = DeterministicTokenQuotaBatchStream(dataset, **kwargs)
    for _ in range(10):
        stream.next_batch()
    shares = stream.sampled_tokens
    assert abs(shares["general"] - shares["short"]) <= 20

    state = stream.state_dict()
    expected = stream.next_batch()
    restored = DeterministicTokenQuotaBatchStream(dataset, **kwargs)
    restored.load_state_dict(state)
    assert restored.next_batch() == expected
