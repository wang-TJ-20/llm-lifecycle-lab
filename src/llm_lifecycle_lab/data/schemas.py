"""Validation for the four JSONL formats used by the first release."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.contracts import JsonContract, RecordKind
from llm_lifecycle_lab.exceptions import DataValidationError

_ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True, slots=True)
class ValidationIssue(JsonContract):
    line: int
    code: str
    message: str
    field: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport(JsonContract):
    source: str
    record_kind: RecordKind
    records_seen: int
    valid_records: int
    issues: tuple[ValidationIssue, ...]
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class ValidatedData:
    records: tuple[dict[str, Any], ...]
    report: ValidationReport


def validate_jsonl(
    path: str | Path,
    record_kind: RecordKind | str,
    *,
    max_errors: int = 100,
) -> ValidatedData:
    """Read JSONL, report deterministic schema errors, and retain valid rows."""

    if max_errors <= 0:
        raise ValueError("max_errors must be positive")

    source = Path(path)
    if not source.is_file():
        raise DataValidationError(f"data file does not exist: {source}")

    kind = RecordKind(record_kind)
    validator = _VALIDATORS[kind]
    records: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []
    seen_ids: dict[str, int] = {}
    records_seen = 0
    truncated = False

    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                records_seen += 1
                try:
                    value = json.loads(
                        raw_line,
                        parse_constant=_reject_non_finite_number,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    issues.append(
                        ValidationIssue(
                            line=line_number,
                            code="invalid_json",
                            message=getattr(exc, "msg", str(exc)),
                        )
                    )
                    if len(issues) >= max_errors:
                        truncated = True
                        break
                    continue

                if not isinstance(value, Mapping):
                    issues.append(
                        ValidationIssue(
                            line=line_number,
                            code="record_not_object",
                            message="each JSONL row must be an object",
                        )
                    )
                    if len(issues) >= max_errors:
                        truncated = True
                        break
                    continue

                record = dict(value)
                row_issues = _validate_common(record, line_number)
                row_issues.extend(validator(record, line_number))
                record_id = record.get("id")
                if isinstance(record_id, str) and record_id.strip():
                    first_line = seen_ids.get(record_id)
                    if first_line is not None:
                        row_issues.append(
                            ValidationIssue(
                                line=line_number,
                                field="id",
                                code="duplicate_id",
                                message=(
                                    f"id {record_id!r} already appeared on "
                                    f"line {first_line}"
                                ),
                            )
                        )
                    else:
                        seen_ids[record_id] = line_number

                if row_issues:
                    issues.extend(row_issues)
                    if len(issues) >= max_errors:
                        issues = issues[:max_errors]
                        truncated = True
                        break
                else:
                    records.append(record)
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"data file is not valid UTF-8: {source}") from exc
    except OSError as exc:
        raise DataValidationError(f"cannot read data file {source}: {exc}") from exc

    if records_seen == 0:
        issues.append(
            ValidationIssue(
                line=0,
                code="empty_dataset",
                message="the input contains no JSON records",
            )
        )

    report = ValidationReport(
        source=str(source),
        record_kind=kind,
        records_seen=records_seen,
        valid_records=len(records),
        issues=tuple(issues),
        truncated=truncated,
    )
    return ValidatedData(records=tuple(records), report=report)


def format_validation_failure(report: ValidationReport) -> str:
    details = "; ".join(
        f"line {issue.line} [{issue.code}] {issue.message}"
        for issue in report.issues[:5]
    )
    suffix = "; additional errors omitted" if report.truncated else ""
    return (
        f"{report.source} failed {report.record_kind.value} validation: "
        f"{len(report.issues)} error(s): {details}{suffix}"
    )


def _validate_common(
    record: Mapping[str, Any],
    line: int,
) -> list[ValidationIssue]:
    return _required_non_empty_strings(record, line, ("id",))


def _validate_pretrain(
    record: Mapping[str, Any],
    line: int,
) -> list[ValidationIssue]:
    return _required_non_empty_strings(record, line, ("text", "source"))


def _validate_sft(
    record: Mapping[str, Any],
    line: int,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return [
            ValidationIssue(
                line=line,
                field="messages",
                code="invalid_messages",
                message="messages must be a non-empty list",
            )
        ]

    assistant_messages = 0
    for index, message in enumerate(messages):
        field = f"messages[{index}]"
        if not isinstance(message, Mapping):
            issues.append(
                ValidationIssue(
                    line=line,
                    field=field,
                    code="invalid_message",
                    message="message must be an object",
                )
            )
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in _ALLOWED_ROLES:
            issues.append(
                ValidationIssue(
                    line=line,
                    field=f"{field}.role",
                    code="invalid_role",
                    message=f"role must be one of {sorted(_ALLOWED_ROLES)}",
                )
            )
        if not isinstance(content, str) or not content.strip():
            issues.append(
                ValidationIssue(
                    line=line,
                    field=f"{field}.content",
                    code="empty_content",
                    message="message content must be a non-empty string",
                )
            )
        if role == "assistant":
            assistant_messages += 1

    if assistant_messages == 0:
        issues.append(
            ValidationIssue(
                line=line,
                field="messages",
                code="missing_assistant",
                message="SFT record must contain an assistant response",
            )
        )
    return issues


def _validate_dpo(
    record: Mapping[str, Any],
    line: int,
) -> list[ValidationIssue]:
    issues = _required_non_empty_strings(
        record,
        line,
        ("prompt", "chosen", "rejected"),
    )
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if (
        isinstance(chosen, str)
        and isinstance(rejected, str)
        and chosen.strip() == rejected.strip()
    ):
        issues.append(
            ValidationIssue(
                line=line,
                field="chosen,rejected",
                code="identical_pair",
                message="chosen and rejected responses must differ",
            )
        )
    return issues


def _validate_grpo(
    record: Mapping[str, Any],
    line: int,
) -> list[ValidationIssue]:
    issues = _required_non_empty_strings(record, line, ("prompt", "answer"))
    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        issues.append(
            ValidationIssue(
                line=line,
                field="metadata",
                code="invalid_metadata",
                message="metadata must be an object when provided",
            )
        )
    return issues


def _required_non_empty_strings(
    record: Mapping[str, Any],
    line: int,
    fields: tuple[str, ...],
) -> list[ValidationIssue]:
    issues = []
    for field in fields:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(
                ValidationIssue(
                    line=line,
                    field=field,
                    code="missing_or_empty",
                    message=f"{field} must be a non-empty string",
                )
            )
    return issues


_VALIDATORS: dict[
    RecordKind,
    Callable[[Mapping[str, Any], int], list[ValidationIssue]],
] = {
    RecordKind.PRETRAIN: _validate_pretrain,
    RecordKind.SFT: _validate_sft,
    RecordKind.DPO: _validate_dpo,
    RecordKind.GRPO: _validate_grpo,
}


def _reject_non_finite_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")
