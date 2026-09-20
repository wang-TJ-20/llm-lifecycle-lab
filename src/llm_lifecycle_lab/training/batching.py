"""Deterministic, resumable batching without hidden DataLoader state."""

from __future__ import annotations

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
