"""Deterministic group-aware dataset splitting."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llm_lifecycle_lab.exceptions import DataValidationError

_SPLIT_NAMES = ("train", "dev", "test")


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.8
    dev: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.dev, self.test)
        if any(value < 0 for value in values):
            raise ValueError("split ratios must be non-negative")
        if self.train <= 0:
            raise ValueError("train ratio must be positive")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1.0")


def split_records(
    records: Sequence[Mapping[str, Any]],
    *,
    ratios: SplitRatios,
    seed: int,
    group_by: str = "id",
) -> dict[str, list[dict[str, Any]]]:
    """Assign complete groups using a stable hash independent of input order."""

    if seed < 0:
        raise ValueError("split seed must be non-negative")
    if not group_by.strip():
        raise ValueError("group_by must not be empty")

    result: dict[str, list[dict[str, Any]]] = {
        split_name: [] for split_name in _SPLIT_NAMES
    }
    group_assignments: dict[str, str] = {}

    for record in records:
        group = group_value(record, group_by)
        split_name = group_assignments.setdefault(
            group,
            _assign_group(group, ratios=ratios, seed=seed),
        )
        result[split_name].append(dict(record))

    leakage = find_split_leakage(result, group_by=group_by)
    if leakage:
        raise DataValidationError(
            "group leakage detected after split: " + ", ".join(leakage[:5])
        )
    return result


def group_value(record: Mapping[str, Any], group_by: str) -> str:
    value: Any = record
    for segment in group_by.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            raise DataValidationError(
                f"record {record.get('id', '<unknown>')!r} is missing "
                f"group field {group_by!r}"
            )
        value = value[segment]
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise DataValidationError(
            f"group field {group_by!r} must be a string or integer"
        )
    normalized = str(value).strip()
    if not normalized:
        raise DataValidationError(f"group field {group_by!r} must not be empty")
    return normalized


def find_split_leakage(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    group_by: str,
) -> list[str]:
    owners: dict[str, str] = {}
    leakage: set[str] = set()
    for split_name, records in splits.items():
        for record in records:
            group = group_value(record, group_by)
            previous = owners.setdefault(group, split_name)
            if previous != split_name:
                leakage.add(group)
    return sorted(leakage)


def _assign_group(
    group: str,
    *,
    ratios: SplitRatios,
    seed: int,
) -> str:
    digest = hashlib.sha256(f"{seed}\0{group}".encode()).digest()
    score = int.from_bytes(digest[:8], byteorder="big") / 2**64
    if score < ratios.train:
        return "train"
    if score < ratios.train + ratios.dev:
        return "dev"
    return "test"
