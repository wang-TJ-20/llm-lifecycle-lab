"""Fail-fast qualification for expanded pretraining corpora."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.contracts import RecordKind
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.prepare import load_data_manifest, verify_data_manifest
from llm_lifecycle_lab.exceptions import ContractError, DataValidationError
from llm_lifecycle_lab.tokenizer import NativeTokenizer


def qualify_pretraining_data(
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    *,
    minimum_train_tokens: int = 300_000_000,
    maximum_train_tokens: int = 500_000_000,
    minimum_language_fraction: float = 0.40,
    maximum_language_fraction: float = 0.60,
    maximum_source_fraction: float = 0.50,
    maximum_synthetic_fraction: float = 0.30,
) -> dict[str, Any]:
    """Count fixed-tokenizer train tokens and enforce the Base-v2 data contract."""

    _validate_thresholds(
        minimum_train_tokens=minimum_train_tokens,
        maximum_train_tokens=maximum_train_tokens,
        minimum_language_fraction=minimum_language_fraction,
        maximum_language_fraction=maximum_language_fraction,
        maximum_source_fraction=maximum_source_fraction,
        maximum_synthetic_fraction=maximum_synthetic_fraction,
    )
    path = Path(manifest_path)
    failures = verify_data_manifest(path)
    if failures:
        raise DataValidationError("; ".join(failures))
    manifest = load_data_manifest(path)
    if manifest.record_kind is not RecordKind.PRETRAIN:
        raise ContractError("pretraining qualification requires a pretrain manifest")
    tokenizer = NativeTokenizer.from_directory(tokenizer_path)
    components = _mixture_components(manifest.source_metadata)
    train = next(split for split in manifest.splits if split.name == "train")

    language_tokens: dict[str, int] = defaultdict(int)
    source_tokens: dict[str, int] = defaultdict(int)
    records = 0
    train_path = path.parent / train.path
    try:
        with train_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                language = row.get("language")
                source_id = row.get("source_recipe_id")
                if not isinstance(text, str) or not text.strip():
                    raise DataValidationError(
                        f"{train_path} line {line_number} has no text"
                    )
                if language not in {"en", "zh"}:
                    raise DataValidationError(
                        f"{train_path} line {line_number} has invalid language"
                    )
                if source_id not in components:
                    raise DataValidationError(
                        f"{train_path} line {line_number} has undeclared "
                        "source_recipe_id"
                    )
                tokens = len(tokenizer.encode(text)) + 2
                language_tokens[str(language)] += tokens
                source_tokens[str(source_id)] += tokens
                records += 1
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid JSON in {train_path}") from exc
    except OSError as exc:
        raise DataValidationError(f"cannot read {train_path}: {exc}") from exc

    if records != train.records:
        raise DataValidationError(
            f"train record count changed: expected {train.records}, got {records}"
        )
    if set(language_tokens) != {"en", "zh"}:
        raise DataValidationError("prepared train split must contain en and zh")
    if set(source_tokens) != set(components):
        missing = sorted(set(components) - set(source_tokens))
        raise DataValidationError(
            "prepared train split omits mixture components: " + ", ".join(missing)
        )
    total_tokens = sum(language_tokens.values())
    synthetic_tokens = sum(
        source_tokens[source_id]
        for source_id, component in components.items()
        if component["synthetic"]
    )
    languages = {
        language: _count_and_fraction(tokens, total_tokens)
        for language, tokens in sorted(language_tokens.items())
    }
    sources = {
        source_id: {
            **_count_and_fraction(tokens, total_tokens),
            "language": components[source_id]["language"],
            "synthetic": components[source_id]["synthetic"],
            "repository": components[source_id]["repository"],
            "revision": components[source_id]["revision"],
        }
        for source_id, tokens in sorted(source_tokens.items())
    }
    synthetic_fraction = synthetic_tokens / total_tokens
    checks = [
        _check(
            "train-token-range",
            minimum_train_tokens <= total_tokens <= maximum_train_tokens,
            total_tokens,
            {"minimum": minimum_train_tokens, "maximum": maximum_train_tokens},
        ),
        *[
            _check(
                f"language-{language}-fraction",
                minimum_language_fraction
                <= values["fraction"]
                <= maximum_language_fraction,
                values["fraction"],
                {
                    "minimum": minimum_language_fraction,
                    "maximum": maximum_language_fraction,
                },
            )
            for language, values in languages.items()
        ],
        _check(
            "maximum-source-fraction",
            max(value["fraction"] for value in sources.values())
            <= maximum_source_fraction,
            max(value["fraction"] for value in sources.values()),
            {"maximum": maximum_source_fraction},
        ),
        _check(
            "synthetic-token-fraction",
            synthetic_fraction <= maximum_synthetic_fraction,
            synthetic_fraction,
            {"maximum": maximum_synthetic_fraction},
        ),
    ]
    return {
        "schema_version": "1.0",
        "ok": all(check["ok"] for check in checks),
        "data_manifest": str(path),
        "data_manifest_sha256": sha256_file(path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer.manifest.content_sha256,
        "tokenizer_source_data_manifest_sha256": (
            tokenizer.manifest.source_data_sha256
        ),
        "train_records": records,
        "train_tokens": total_tokens,
        "languages": languages,
        "sources": sources,
        "synthetic_tokens": synthetic_tokens,
        "synthetic_fraction": synthetic_fraction,
        "checks": checks,
    }


def _mixture_components(
    source_metadata: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    values = source_metadata.get("components")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ContractError(
            "pretraining qualification requires mixture component provenance"
        )
    components: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ContractError("mixture component provenance must be mappings")
        recipe_id = value.get("recipe_id")
        language = value.get("language")
        repository = value.get("repository")
        revision = value.get("revision")
        synthetic = value.get("synthetic")
        if (
            not isinstance(recipe_id, str)
            or not recipe_id
            or language not in {"en", "zh"}
            or not isinstance(repository, str)
            or not repository
            or not isinstance(revision, str)
            or len(revision) != 40
            or type(synthetic) is not bool
        ):
            raise ContractError("mixture component provenance is incomplete")
        if recipe_id in components:
            raise ContractError(f"duplicate mixture component: {recipe_id}")
        components[recipe_id] = {
            "language": language,
            "repository": repository,
            "revision": revision,
            "synthetic": synthetic,
        }
    if len(components) < 2:
        raise ContractError("pretraining qualification requires multiple sources")
    return components


def _validate_thresholds(
    *,
    minimum_train_tokens: int,
    maximum_train_tokens: int,
    minimum_language_fraction: float,
    maximum_language_fraction: float,
    maximum_source_fraction: float,
    maximum_synthetic_fraction: float,
) -> None:
    if (
        minimum_train_tokens <= 0
        or maximum_train_tokens < minimum_train_tokens
    ):
        raise ValueError("invalid train-token range")
    fractions = (
        minimum_language_fraction,
        maximum_language_fraction,
        maximum_source_fraction,
        maximum_synthetic_fraction,
    )
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in fractions):
        raise ValueError("qualification fractions must be finite values in [0, 1]")
    if maximum_language_fraction < minimum_language_fraction:
        raise ValueError("invalid language-fraction range")


def _count_and_fraction(count: int, total: int) -> dict[str, float | int]:
    return {"tokens": count, "fraction": count / total}


def _check(
    name: str,
    ok: bool,
    actual: int | float,
    expected: Mapping[str, int | float],
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "actual": actual,
        "expected": dict(expected),
    }
