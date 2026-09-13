"""Bounded memory probes and streaming, normalized substring overlap checks."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.contracts import RecordKind
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.prepare import load_data_manifest, verify_data_manifest
from llm_lifecycle_lab.evaluation.suite import digest
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.tokenizer import NativeTokenizer

CORPUS_PROTOCOL = {
    "normalization": "NFKC-whitespace-collapse-v1",
    "overlap_min_chars": 24,
    "memory_prefix_tokens": 32,
    "memory_suffix_tokens": 12,
    "memory_per_language_split": 2,
    "sampling": "first-eligible-in-manifest-order",
}


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def case_texts(suite: dict[str, Any]) -> dict[str, str]:
    return {
        case["id"]: (
            case["text"]
            if case["kind"] == "corpus"
            else " ".join(turn["prompt"] for turn in case["turns"])
            if case["kind"] == "multiturn"
            else case["prompt"]
        )
        for case in suite["cases"]
    }


def scan_corpus(
    manifest_path: str | Path,
    *,
    suite: dict[str, Any],
    tokenizer: NativeTokenizer,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = load_data_manifest(path)
    if manifest.record_kind is not RecordKind.PRETRAIN:
        raise ContractError("memory corpus must be a pretrain Data Manifest")
    if sha256_file(path) != tokenizer.manifest.source_data_sha256:
        raise ContractError("memory corpus is not bound to this Base tokenizer")
    failures = verify_data_manifest(path)
    if failures:
        raise ContractError("; ".join(failures))
    queries = {
        key: normalize(value)
        for key, value in case_texts(suite).items()
        if len(normalize(value)) >= CORPUS_PROTOCOL["overlap_min_chars"]
    }
    hits: set[str] = set()
    train_hashes: set[str] = set()
    heldout_hashes: list[str] = []
    probes: list[dict[str, Any]] = []
    scanned = {}
    per_bucket: dict[tuple[str, str], int] = {}
    prefix_length = CORPUS_PROTOCOL["memory_prefix_tokens"]
    suffix_length = CORPUS_PROTOCOL["memory_suffix_tokens"]
    # The complete train split is scanned; probe generation alone is bounded.
    for split_name in ("train", "dev", "test"):
        split = next(item for item in manifest.splits if item.name == split_name)
        count = 0
        with (path.parent / split.path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("text"), str) or not row["text"].strip():
                    raise ContractError("corpus row has no usable text")
                text = normalize(row["text"])
                text_hash = digest({"text": text})
                count += 1
                if split_name == "train":
                    train_hashes.add(text_hash)
                    hits.update(key for key, query in queries.items() if query in text)
                else:
                    heldout_hashes.append(text_hash)
                language = row.get("language")
                key = (split_name, language)
                if (
                    split_name == "dev"
                    or language not in {"en", "zh"}
                    or per_bucket.get(key, 0)
                    >= CORPUS_PROTOCOL["memory_per_language_split"]
                ):
                    continue
                tokens = tokenizer.encode(row["text"])
                if len(tokens) < prefix_length + suffix_length:
                    continue
                probe_index = per_bucket.get(key, 0)
                probes.append(
                    {
                        "id": f"memory-{split_name}-{language}-{probe_index}",
                        "language": language,
                        "split": split_name,
                        "source_id_sha256": digest({"id": str(row.get("id", ""))}),
                        "text_sha256": text_hash,
                        "prefix": [tokenizer.bos_token_id, *tokens[:prefix_length]],
                        "target": tokens[prefix_length : prefix_length + suffix_length],
                    }
                )
                per_bucket[key] = per_bucket.get(key, 0) + 1
        scanned[split_name] = count
    for probe in probes:
        probe["exact_train_overlap"] = probe["text_sha256"] in train_hashes
    return {
        "status": "checked",
        "manifest_sha256": sha256_file(path),
        "dataset_id": manifest.dataset_id,
        "license": manifest.license,
        "scanned_records": scanned,
        "eligible_queries": len(queries),
        "matched_query_ids": sorted(hits),
        "excluded_short_queries": len(case_texts(suite)) - len(queries),
        "heldout_exact_train_matches": sum(h in train_hashes for h in heldout_hashes),
        "heldout_records": len(heldout_hashes),
        "probes": probes,
        "limitations": (
            "Normalized full-text and query-substring checks only; no semantic or "
            "fuzzy detection. Short queries are excluded, not counted as clean. "
            "Membership refers to the supplied tokenizer-bound corpus, not proof "
            "that any particular optimizer step observed the selected document."
        ),
    }
