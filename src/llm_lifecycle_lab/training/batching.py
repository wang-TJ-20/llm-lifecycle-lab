"""Deterministic, resumable batching without hidden DataLoader state."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from llm_lifecycle_lab.exceptions import ContractError


@dataclass(frozen=True, slots=True)
class BatchStreamState:
    epoch: int
    offset: int


class DeterministicBatchStream:
    """Shuffle by epoch seed and persist the exact next-example offset."""

    def __init__(
        self,
        dataset: Sequence[Any],
        *,
        batch_size: int,
        seed: int,
        collate_fn: Callable[[list[Any]], Any],
    ) -> None:
        if len(dataset) == 0:
            raise ContractError("batch stream requires a non-empty dataset")
        if batch_size <= 0:
            raise ContractError("batch_size must be positive")
        if seed < 0:
            raise ContractError("batch stream seed must be non-negative")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.collate_fn = collate_fn
        self.epoch = 0
        self.offset = 0
        self._permutation = self._permutation_for_epoch(0)

    def next_batch(self) -> Any:
        if self.offset >= len(self._permutation):
            self.epoch += 1
            self.offset = 0
            self._permutation = self._permutation_for_epoch(self.epoch)

        end = min(self.offset + self.batch_size, len(self._permutation))
        indices = self._permutation[self.offset : end]
        self.offset = end
        return self.collate_fn([self.dataset[index] for index in indices])

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state: dict[str, int]) -> None:
        epoch = int(state["epoch"])
        offset = int(state["offset"])
        if epoch < 0 or not 0 <= offset <= len(self.dataset):
            raise ContractError("invalid batch stream state")
        self.epoch = epoch
        self.offset = offset
        self._permutation = self._permutation_for_epoch(epoch)

    def _permutation_for_epoch(self, epoch: int) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        return torch.randperm(len(self.dataset), generator=generator).tolist()


class DeterministicTokenQuotaBatchStream:
    """Sample strata so cumulative supervised-token shares follow fixed quotas."""

    def __init__(
        self,
        dataset: Sequence[dict[str, Any]],
        *,
        batch_size: int,
        seed: int,
        collate_fn: Callable[[list[Any]], Any],
        stratum_field: str,
        token_count_field: str,
        weights: dict[str, float],
    ) -> None:
        if not dataset:
            raise ContractError("batch stream requires a non-empty dataset")
        if batch_size <= 0:
            raise ContractError("batch_size must be positive")
        if seed < 0:
            raise ContractError("batch stream seed must be non-negative")
        if not weights or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in weights.values()
        ):
            raise ContractError("token quota weights must be finite and positive")

        buckets: dict[str, list[int]] = {name: [] for name in weights}
        token_counts = []
        for index, example in enumerate(dataset):
            stratum = example.get(stratum_field)
            if stratum not in buckets:
                raise ContractError(
                    f"training example has unsupported {stratum_field}: {stratum!r}"
                )
            count = example.get(token_count_field)
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
            ):
                raise ContractError(
                    f"training example has invalid {token_count_field}"
                )
            buckets[stratum].append(index)
            token_counts.append(count)
        if any(not bucket for bucket in buckets.values()):
            raise ContractError("token quota stream requires every configured stratum")

        total_weight = sum(float(value) for value in weights.values())
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.collate_fn = collate_fn
        self.stratum_field = stratum_field
        self.token_count_field = token_count_field
        self.strata = tuple(weights)
        self.weights = {
            name: float(weights[name]) / total_weight for name in self.strata
        }
        self.buckets = buckets
        self.token_counts = token_counts
        self.epochs = {name: 0 for name in self.strata}
        self.offsets = {name: 0 for name in self.strata}
        self.sampled_tokens = {name: 0 for name in self.strata}
        self.samples_seen = 0
        self._permutations = {
            name: self._permutation_for_epoch(name, 0) for name in self.strata
        }

    def next_batch(self) -> Any:
        examples = []
        for _ in range(self.batch_size):
            stratum = min(
                self.strata,
                key=lambda name: (
                    self.sampled_tokens[name] / self.weights[name],
                    self.strata.index(name),
                ),
            )
            if self.offsets[stratum] >= len(self._permutations[stratum]):
                self.epochs[stratum] += 1
                self.offsets[stratum] = 0
                self._permutations[stratum] = self._permutation_for_epoch(
                    stratum,
                    self.epochs[stratum],
                )
            index = self._permutations[stratum][self.offsets[stratum]]
            self.offsets[stratum] += 1
            self.sampled_tokens[stratum] += self.token_counts[index]
            self.samples_seen += 1
            examples.append(self.dataset[index])
        return self.collate_fn(examples)

    def state_dict(self) -> dict[str, Any]:
        return {
            "strategy": "supervised-token-quota",
            "epochs": dict(self.epochs),
            "offsets": dict(self.offsets),
            "sampled_tokens": dict(self.sampled_tokens),
            "samples_seen": self.samples_seen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("strategy") != "supervised-token-quota":
            raise ContractError("invalid token quota batch stream strategy")
        try:
            epochs = {name: int(state["epochs"][name]) for name in self.strata}
            offsets = {name: int(state["offsets"][name]) for name in self.strata}
            sampled_tokens = {
                name: int(state["sampled_tokens"][name]) for name in self.strata
            }
            samples_seen = int(state["samples_seen"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("invalid token quota batch stream state") from exc
        if (
            set(state["epochs"]) != set(self.strata)
            or set(state["offsets"]) != set(self.strata)
            or set(state["sampled_tokens"]) != set(self.strata)
            or samples_seen < 0
            or any(value < 0 for value in epochs.values())
            or any(value < 0 for value in sampled_tokens.values())
            or any(
                not 0 <= offsets[name] <= len(self.buckets[name])
                for name in self.strata
            )
        ):
            raise ContractError("invalid token quota batch stream state")
        self.epochs = epochs
        self.offsets = offsets
        self.sampled_tokens = sampled_tokens
        self.samples_seen = samples_seen
        self._permutations = {
            name: self._permutation_for_epoch(name, epochs[name])
            for name in self.strata
        }

    def _permutation_for_epoch(self, stratum: str, epoch: int) -> list[int]:
        digest = hashlib.sha256(stratum.encode("utf-8")).digest()
        stratum_seed = int.from_bytes(digest[:8], "big")
        generator = torch.Generator()
        generator.manual_seed((self.seed + stratum_seed + epoch) % (2**63 - 1))
        order = torch.randperm(
            len(self.buckets[stratum]),
            generator=generator,
        ).tolist()
        return [self.buckets[stratum][index] for index in order]


def stable_stratified_subset(
    dataset: Sequence[dict[str, Any]],
    *,
    limit: int,
    strata: Sequence[Any],
    stratum_field: str,
    identity_field: str,
) -> list[dict[str, Any]]:
    """Select a bounded, interleaved subset independently of source order."""

    if limit < len(strata):
        raise ContractError("subset limit must cover every requested stratum")
    buckets: dict[Any, list[dict[str, Any]]] = {value: [] for value in strata}
    for example in dataset:
        value = example.get(stratum_field)
        if value not in buckets:
            raise ContractError(
                f"evaluation example has unsupported {stratum_field}: {value!r}"
            )
        identity = example.get(identity_field)
        if not isinstance(identity, str) or not identity:
            raise ContractError(f"evaluation example has no usable {identity_field}")
        buckets[value].append(example)
    if any(not bucket for bucket in buckets.values()):
        raise ContractError("evaluation subset must contain every requested stratum")
    for bucket in buckets.values():
        bucket.sort(
            key=lambda example: hashlib.sha256(
                example[identity_field].encode("utf-8")
            ).digest()
        )

    selected = []
    offset = 0
    while len(selected) < limit:
        added = False
        for value in strata:
            bucket = buckets[value]
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected
