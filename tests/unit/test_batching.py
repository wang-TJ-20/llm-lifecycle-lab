from __future__ import annotations

import pytest

from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.training.batching import stable_stratified_subset


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
