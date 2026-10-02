from __future__ import annotations

import pytest

from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.training.batching import (
    DeterministicTokenQuotaBatchStream,
    stable_stratified_subset,
)


def _dataset() -> list[dict[str, object]]:
    return [
        {
            "id": f"{stratum}-{index}",
            "stratum": stratum,
            "tokens": tokens,
        }
        for stratum, counts in (("short", (1, 2, 3)), ("long", (8, 10)))
        for index, tokens in enumerate(counts)
    ]


def _stream() -> DeterministicTokenQuotaBatchStream:
    return DeterministicTokenQuotaBatchStream(
        _dataset(),
        batch_size=3,
        seed=42,
        collate_fn=lambda rows: [row["id"] for row in rows],
        stratum_field="stratum",
        token_count_field="tokens",
        weights={"short": 0.25, "long": 0.75},
    )


def test_token_quota_stream_resume_is_exact() -> None:
    uninterrupted = _stream()
    prefix = [uninterrupted.next_batch() for _ in range(4)]
    state = uninterrupted.state_dict()
    suffix = [uninterrupted.next_batch() for _ in range(6)]

    resumed = _stream()
    resumed.load_state_dict(state)

    assert prefix
    assert [resumed.next_batch() for _ in range(6)] == suffix
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_token_quota_stream_is_deterministic_and_tracks_token_shares() -> None:
    first = _stream()
    second = _stream()

    first_batches = [first.next_batch() for _ in range(100)]
    second_batches = [second.next_batch() for _ in range(100)]

    assert first_batches == second_batches
    total = sum(first.sampled_tokens.values())
    assert first.sampled_tokens["short"] / total == pytest.approx(0.25, abs=0.04)
    assert first.sampled_tokens["long"] / total == pytest.approx(0.75, abs=0.04)


def test_token_quota_stream_rejects_incomplete_strata() -> None:
    with pytest.raises(ContractError, match="every configured stratum"):
        DeterministicTokenQuotaBatchStream(
            _dataset(),
            batch_size=1,
            seed=42,
            collate_fn=lambda rows: rows,
            stratum_field="stratum",
            token_count_field="tokens",
            weights={"short": 0.5, "long": 0.25, "missing": 0.25},
        )


def test_stable_subset_is_balanced_and_source_order_independent() -> None:
    rows = [
        {"id_hash": f"{language}-{index}", "language": language}
        for language in ("en", "zh")
        for index in range(5)
    ]
    first = stable_stratified_subset(
        rows,
        limit=6,
        strata=("en", "zh"),
        stratum_field="language",
        identity_field="id_hash",
    )
    second = stable_stratified_subset(
        list(reversed(rows)),
        limit=6,
        strata=("en", "zh"),
        stratum_field="language",
        identity_field="id_hash",
    )

    assert [row["id_hash"] for row in first] == [row["id_hash"] for row in second]
    assert [row["language"] for row in first] == ["en", "zh"] * 3
