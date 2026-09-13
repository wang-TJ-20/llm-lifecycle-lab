"""Leakage-aware preference pairs encoded with response-only likelihood masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab.contracts import RecordKind
from llm_lifecycle_lab.data.prepare import load_data_manifest, verify_data_manifest
from llm_lifecycle_lab.data.schemas import format_validation_failure, validate_jsonl
from llm_lifecycle_lab.data.sft import encode_sft_example
from llm_lifecycle_lab.data.split import group_value
from llm_lifecycle_lab.evaluation.corpus import normalize
from llm_lifecycle_lab.evaluation.suite import digest, load_suite
from llm_lifecycle_lab.exceptions import ContractError, DataValidationError


def encode_dpo_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    *,
    tokenizer: Any,
    sequence_length: int,
    language: str,
    pair_id: str,
) -> dict[str, Any]:
    if normalize(chosen) == normalize(rejected):
        raise ContractError("DPO chosen and rejected responses must differ")
    examples = {}
    for name, response in (("chosen", chosen), ("rejected", rejected)):
        encoded = encode_sft_example(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            language=language,
        )
        examples.update({f"{name}_{key}": value for key, value in encoded.items()})
    return {
        "pair_id_sha256": digest({"id": pair_id}),
        "language": language,
        **examples,
    }


def collate_dpo(examples: list[dict[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    if not examples:
        raise ContractError("cannot collate an empty DPO batch")
    result: dict[str, Any] = {
        "pair_id_sha256": [example["pair_id_sha256"] for example in examples]
    }
    padding = {
        "input_ids": pad_token_id,
        "labels": -100,
        "attention_mask": False,
        "language_ids": -1,
        "byte_weights": 0.0,
    }
    for choice in ("chosen", "rejected"):
        length = max(example[f"{choice}_input_ids"].numel() for example in examples)
        for field, pad in padding.items():
            key = f"{choice}_{field}"
            result[key] = torch.stack(
                [
                    torch.cat(
                        (
                            example[key],
                            torch.full(
                                (length - example[key].numel(),),
                                pad,
                                dtype=example[key].dtype,
                            ),
                        )
                    )
                    for example in examples
                ]
            )
    for field in ("reference_chosen_logps", "reference_rejected_logps"):
        if all(field in example for example in examples):
            result[field] = torch.tensor(
                [float(example[field]) for example in examples], dtype=torch.float32
            )
    return result


def load_dpo_splits(
    manifest_path: str | Path,
    *,
    tokenizer: Any,
    sequence_length: int,
    evaluation_suite: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    path = Path(manifest_path)
    manifest = load_data_manifest(path)
    if manifest.record_kind is not RecordKind.DPO:
        raise ContractError("DPO requires a DPO Data Manifest")
    failures = verify_data_manifest(path)
    if failures:
        raise ContractError("; ".join(failures))
    suite = load_suite(evaluation_suite)
    forbidden_groups = {case["group"] for case in suite["cases"]}
    forbidden_text = {
        normalize(turn["prompt"])
        for case in suite["cases"]
        if case["kind"] != "corpus"
        for turn in (case["turns"] if case["kind"] == "multiturn" else [case])
    }
    groups: dict[str, str] = {}
    pairs: set[str] = set()
    splits = {}
    summary = {"suite_sha256": digest(suite), "splits": {}}
    for split in manifest.splits:
        validated = validate_jsonl(path.parent / split.path, RecordKind.DPO)
        if not validated.report.ok:
            raise DataValidationError(format_validation_failure(validated.report))
        examples = []
        language_counts = {"en": 0, "zh": 0}
        for row in validated.records:
            language = row.get("language")
            if language not in language_counts:
                raise ContractError("DPO examples must declare language=en or zh")
            group = group_value(row, manifest.group_by)
            if group in groups and groups[group] != split.name:
                raise ContractError("DPO source/template group leaks across splits")
            groups[group] = split.name
            pair_hash = digest(
                {
                    "prompt": normalize(row["prompt"]),
                    "chosen": normalize(row["chosen"]),
                    "rejected": normalize(row["rejected"]),
                }
            )
            if pair_hash in pairs:
                raise ContractError("duplicate normalized DPO pair")
            pairs.add(pair_hash)
            if (
                group in forbidden_groups
                or row.get("template_id") in forbidden_groups
                or normalize(row["prompt"]) in forbidden_text
            ):
                raise ContractError("DPO data overlaps the fixed evaluation suite")
            examples.append(
                encode_dpo_pair(
                    row["prompt"],
                    row["chosen"],
                    row["rejected"],
                    tokenizer=tokenizer,
                    sequence_length=sequence_length,
                    language=language,
                    pair_id=row["id"],
                )
            )
            language_counts[language] += 1
        if not examples or not all(language_counts.values()):
            raise ContractError(f"DPO {split.name} must contain both en and zh")
        splits[split.name] = examples
        summary["splits"][split.name] = {
            "pairs": len(examples),
            "languages": language_counts,
            "response_tokens": sum(
                int((example[f"{choice}_labels"][1:] != -100).sum())
                for example in examples
                for choice in ("chosen", "rejected")
            ),
        }
    return splits, summary
