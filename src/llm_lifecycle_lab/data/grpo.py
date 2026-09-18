"""Prompt groups and deterministic, executable-free reward verifiers for GRPO."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.config import canonical_json
from llm_lifecycle_lab.contracts import RecordKind
from llm_lifecycle_lab.data.prepare import load_data_manifest, verify_data_manifest
from llm_lifecycle_lab.data.schemas import format_validation_failure, validate_jsonl
from llm_lifecycle_lab.data.split import group_value
from llm_lifecycle_lab.evaluation.corpus import normalize
from llm_lifecycle_lab.evaluation.suite import digest, load_suite
from llm_lifecycle_lab.exceptions import ContractError, DataValidationError

VERIFIERS = {"exact", "integer", "json"}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _parse_json(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def evaluate_verifier_response(
    text: str,
    answer: str,
    verifier: str,
) -> tuple[float, bool]:
    """Return deterministic reward and whether the candidate parsed."""

    if verifier == "exact":
        return float(text.strip() == answer), True
    if verifier == "integer":
        pattern = re.compile(r"[+-]?\d+")
        candidate = text.strip()
        parsed = pattern.fullmatch(candidate) is not None
        if not parsed or not pattern.fullmatch(answer):
            return 0.0, parsed
        return float(int(candidate) == int(answer)), True
    if verifier == "json":
        try:
            actual = _parse_json(text)
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
            return 0.0, False
        try:
            expected = _parse_json(answer)
            actual_json = canonical_json(actual)
            expected_json = canonical_json(expected)
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
            return 0.0, True
        return (
            float(
                isinstance(actual, (dict, list))
                and type(actual) is type(expected)
                and actual_json == expected_json
            ),
            True,
        )
    raise ContractError(f"unsupported GRPO verifier: {verifier}")


def reward_response(text: str, answer: str, verifier: str) -> float:
    return evaluate_verifier_response(text, answer, verifier)[0]


def rollout_seed(example_id: str, group_index: int) -> int:
    if group_index < 0:
        raise ContractError("group_index must be non-negative")
    return int(digest({"id": example_id, "group_index": group_index})[:8], 16)


def collate_grpo(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if not examples:
        raise ContractError("cannot collate an empty GRPO batch")
    return {
        key: [example[key] for example in examples]
        for key in (
            "example_id",
            "prompt",
            "prompt_ids",
            "answer",
            "verifier",
            "language",
        )
    }


def load_grpo_splits(
    manifest_path: str | Path,
    *,
    tokenizer: Any,
    sequence_length: int,
    max_new_tokens: int,
    evaluation_suite: str | Path,
    split_names: Sequence[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    path = Path(manifest_path)
    manifest = load_data_manifest(path)
    if manifest.record_kind is not RecordKind.GRPO:
        raise ContractError("GRPO requires a GRPO Data Manifest")
    failures = verify_data_manifest(path)
    if failures:
        raise ContractError("; ".join(failures))
    suite = load_suite(evaluation_suite)
    forbidden_groups = {case["group"] for case in suite["cases"]}
    forbidden_prompts = {
        normalize(turn["prompt"])
        for case in suite["cases"]
        if case["kind"] != "corpus"
        for turn in (case["turns"] if case["kind"] == "multiturn" else [case])
    }
    groups: dict[str, str] = {}
    prompts: set[str] = set()
    splits = {}
    summary = {"suite_sha256": digest(suite), "splits": {}, "verifiers": {}}
    selected_names = (
        {split.name for split in manifest.splits}
        if split_names is None
        else set(split_names)
    )
    available_names = {split.name for split in manifest.splits}
    if not selected_names or not selected_names <= available_names:
        raise ContractError("GRPO split_names must select available splits")
    for split in manifest.splits:
        if split.name not in selected_names:
            continue
        validated = validate_jsonl(path.parent / split.path, RecordKind.GRPO)
        if not validated.report.ok:
            raise DataValidationError(format_validation_failure(validated.report))
        examples = []
        language_counts = {"en": 0, "zh": 0}
        verifier_counts = {name: 0 for name in sorted(VERIFIERS)}
        for row in validated.records:
            language = row.get("language")
            if language not in language_counts:
                raise ContractError("GRPO examples must declare language=en or zh")
            metadata = row.get("metadata", {})
            verifier = metadata.get("verifier") if isinstance(metadata, dict) else None
            if verifier not in VERIFIERS:
                raise ContractError(
                    f"GRPO metadata.verifier must be one of {sorted(VERIFIERS)}"
                )
            # Validate the declared answer before any rollout can consume it.
            if verifier == "integer" and not re.fullmatch(r"[+-]?\d+", row["answer"]):
                raise ContractError("integer verifier requires an integer answer")
            if verifier == "json":
                try:
                    expected = _parse_json(row["answer"])
                    canonical_json(expected)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ContractError("JSON verifier has an invalid answer") from exc
                if not isinstance(expected, (dict, list)):
                    raise ContractError(
                        "JSON verifier answer must be an object or array"
                    )
            group = group_value(row, manifest.group_by)
            if group in groups and groups[group] != split.name:
                raise ContractError("GRPO source/template group leaks across splits")
            groups[group] = split.name
            prompt = normalize(row["prompt"])
            if prompt in prompts:
                raise ContractError("duplicate normalized GRPO prompt")
            prompts.add(prompt)
            if (
                group in forbidden_groups
                or row.get("template_id") in forbidden_groups
                or prompt in forbidden_prompts
            ):
                raise ContractError("GRPO data overlaps the fixed evaluation suite")
            prompt_ids = tokenizer.encode_chat(
                [{"role": "user", "content": row["prompt"]}],
                add_generation_prompt=True,
            )
            if len(prompt_ids) + max_new_tokens > sequence_length:
                raise ContractError(
                    "GRPO prompt plus rollout budget exceeds sequence_length"
                )
            examples.append(
                {
                    "example_id": row["id"],
                    "prompt": row["prompt"],
                    "prompt_ids": prompt_ids,
                    "answer": row["answer"],
                    "verifier": verifier,
                    "language": language,
                    "source_id": str(group),
                }
            )
            language_counts[language] += 1
            verifier_counts[verifier] += 1
        if not examples or not all(language_counts.values()):
            raise ContractError(f"GRPO {split.name} must contain both en and zh")
        splits[split.name] = examples
        summary["splits"][split.name] = {
            "prompts": len(examples),
            "languages": language_counts,
            "verifiers": verifier_counts,
        }
        for name, count in verifier_counts.items():
            summary["verifiers"][name] = summary["verifiers"].get(name, 0) + count
    return splits, summary
