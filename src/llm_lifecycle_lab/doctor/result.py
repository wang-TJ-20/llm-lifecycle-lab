"""Structured diagnostics returned by llmlab doctor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from llm_lifecycle_lab.contracts import JsonContract, utc_now


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class CheckResult(JsonContract):
    name: str
    status: CheckStatus
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DoctorReport(JsonContract):
    profile: str
    checks: tuple[CheckResult, ...]
    created_at: str = field(default_factory=utc_now)

    @property
    def has_failures(self) -> bool:
        return any(check.status is CheckStatus.FAIL for check in self.checks)

    @property
    def exit_code(self) -> int:
        return 1 if self.has_failures else 0

    @property
    def counts(self) -> dict[str, int]:
        return {
            status.value: sum(check.status is status for check in self.checks)
            for status in CheckStatus
        }

    def to_dict(self) -> dict[str, Any]:
        value = JsonContract.to_dict(self)
        value["counts"] = self.counts
        value["ok"] = not self.has_failures
        return value
