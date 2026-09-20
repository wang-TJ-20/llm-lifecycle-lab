"""Assistant-only supervision with explicit split and evaluation-set isolation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab.contracts import RecordKind
from llm_lifecycle_lab.data.prepare import load_data_manifest, verify_data_manifest
from llm_lifecycle_lab.data.schemas import format_validation_failure, validate_jsonl
from llm_lifecycle_lab.data.split import group_value
from llm_lifecycle_lab.evaluation.corpus import normalize
from llm_lifecycle_lab.evaluation.suite import digest, load_suite
from llm_lifecycle_lab.exceptions import ContractError, DataValidationError
from llm_lifecycle_lab.tokenizer import NativeTokenizer

_SHORT_QA_TASKS = {
    "closed_qa",
    "information_extraction",
    "nlpcc_dbqa",
    "short_qa",
}


def encode_sft_example(
    messages: list[dict[str, str]],
    *,
    tokenizer: NativeTokenizer,
    sequence_length: int,
    language: str,
) -> dict[str, torch.Tensor]:
    if language not in {"en", "zh"}:
        raise ContractError("SFT examples must declare language=en or zh")
    ids = tokenizer.encode_chat(messages)
    if messages[-1]["role"] != "assistant":
        raise ContractError("SFT conversation must end with assistant")
    if len(ids) > sequence_length:
        raise ContractError(
            f"SFT example has {len(ids)} tokens, exceeds {sequence_length}; "
            "no implicit truncation is allowed"
        )
    labels = [-100] * len(ids)
    starts = [
        i for i, token in enumerate(ids) if token == tokenizer.chat_start_token_id
    ]
    ends = [i for i, token in enumerate(ids) if token == tokenizer.chat_end_token_id]
    if len(starts) != len(messages) or len(ends) != len(messages):
        raise ContractError("chat control token boundaries do not match messages")
    for message, start, end in zip(messages, starts, ends, strict=True):
        header = tokenizer.encode(message["role"] + "\n")
        body = start + 1 + len(header)
        # Fail rather than supervise part of a role/header when BPE merges across
        # the boundary (possible with unusual leading whitespace in content).
        if ids[start + 1 : body] != header or body >= end:
            raise ContractError("SFT header/body token boundary is ambiguous or empty")
        if message["role"] == "assistant":
            labels[body : end + 1] = ids[body : end + 1]
    if not any(value != -100 for value in labels[1:]):
        raise ContractError("SFT example has zero supervised next-token targets")
    length = len(ids)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(length, dtype=torch.bool),
        "language_ids": torch.full(
            (length,), 0 if language == "en" else 1, dtype=torch.int8
        ),
        # SFT language losses are meaningful; whole-conversation BPB is not.
        "byte_weights": torch.zeros(length),
    }


def collate_sft(
    examples: list[dict[str, torch.Tensor]], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ContractError("cannot collate an empty SFT batch")
    length = max(example["input_ids"].numel() for example in examples)
    padding = {
        "input_ids": pad_token_id,
        "labels": -100,
        "attention_mask": False,
        "language_ids": -1,
        "byte_weights": 0.0,
    }
    result = {}
    for field, pad in padding.items():
        result[field] = torch.stack(
            [
                torch.cat(
                    (
                        example[field],
                        torch.full(
                            (length - example[field].numel(),),
                            pad,
                            dtype=example[field].dtype,
                        ),
                    )
                )
                for example in examples
            ]
        )
    return result


def load_sft_splits(
    manifest_path: str | Path,
    *,
    tokenizer: NativeTokenizer,
    sequence_length: int,
    evaluation_suite: str | Path,
) -> tuple[dict[str, list[dict[str, torch.Tensor]]], dict[str, Any]]:
    path = Path(manifest_path)
    manifest = load_data_manifest(path)
    if manifest.record_kind is not RecordKind.SFT:
        raise ContractError("SFT requires an SFT Data Manifest")
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
    forbidden_passages = {
        normalize(case["text"]) for case in suite["cases"] if case["kind"] == "corpus"
    } | {prompt for prompt in forbidden_prompts if len(prompt) >= 24}
    groups: dict[str, str] = {}
    dialogues: dict[str, str] = {}
    splits = {}
    summary = {"suite_sha256": digest(suite), "splits": {}}
    for split in manifest.splits:
        validated = validate_jsonl(path.parent / split.path, RecordKind.SFT)
        if not validated.report.ok:
            raise DataValidationError(format_validation_failure(validated.report))
        examples = []
        language_counts = {"en": 0, "zh": 0}
        task_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"examples": 0, "supervised_tokens": 0}
        )
        stratum_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"examples": 0, "supervised_tokens": 0}
        )
        for row in validated.records:
            if "template_id" in row and not isinstance(row["template_id"], str):
                raise ContractError("SFT template_id must be text")
            group = group_value(row, manifest.group_by)
            if group in groups and groups[group] != split.name:
                raise ContractError("SFT source/template group leaks across splits")
            groups[group] = split.name
            dialogue_hash = digest(
                {
                    "messages": [
                        {**m, "content": normalize(m["content"])}
                        for m in row["messages"]
                    ]
                }
            )
            if dialogue_hash in dialogues:
                raise ContractError("duplicate normalized SFT conversation")
            dialogues[dialogue_hash] = split.name
            if (
                group in forbidden_groups
                or row.get("template_id") in forbidden_groups
                or any(
                    normalize(m["content"]) in forbidden_prompts
                    for m in row["messages"]
                    if m["role"] == "user"
                )
                or any(
                    passage in normalize(message["content"])
                    for message in row["messages"]
                    for passage in forbidden_passages
                )
            ):
                raise ContractError("SFT data overlaps the fixed evaluation suite")
            example = encode_sft_example(
                row["messages"],
                tokenizer=tokenizer,
                sequence_length=sequence_length,
                language=row.get("language", ""),
            )
            example["example_id_sha256"] = digest({"id": row["id"]})
            example["language"] = row["language"]
            example["task_family"] = _task_family(row)
            example["sampling_stratum"] = (
                f"{example['task_family']}.{row['language']}"
            )
            supervised_tokens = int((example["labels"][1:] != -100).sum())
            example["supervised_token_count"] = supervised_tokens
            examples.append(example)
            language_counts[row["language"]] += 1
            task_counts[example["task_family"]]["examples"] += 1
            task_counts[example["task_family"]][
                "supervised_tokens"
            ] += supervised_tokens
            stratum_counts[example["sampling_stratum"]]["examples"] += 1
            stratum_counts[example["sampling_stratum"]][
                "supervised_tokens"
            ] += supervised_tokens
        if not examples or not all(language_counts.values()):
            raise ContractError(f"SFT {split.name} must contain both en and zh")
        splits[split.name] = examples
        summary["splits"][split.name] = {
            "examples": len(examples),
            "languages": language_counts,
            "supervised_tokens": sum(
                example["supervised_token_count"] for example in examples
            ),
            "task_families": dict(sorted(task_counts.items())),
            "sampling_strata": dict(sorted(stratum_counts.items())),
        }
    # JSON conversion checks the summary before it becomes a training artifact.
    json.dumps(summary, allow_nan=False)
    return splits, summary


def _task_family(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    values = metadata if isinstance(metadata, Mapping) else {}
    task = str(values.get("task", "")).strip()
    transform = str(values.get("transform", "")).strip()
    if task == "classification":
        return "classification"
    if task in _SHORT_QA_TASKS:
        return "short_qa"
    if task == "structured":
        return "structured"
    if task in {"math-verifiable", "numeric"} or transform == (
        "msvamp-parallel-integer-v1"
    ):
        return "numeric"
    return "general"
