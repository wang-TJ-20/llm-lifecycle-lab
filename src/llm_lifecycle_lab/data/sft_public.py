"""Materialize the frozen public SFT source without fetching later-stage data."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.config import load_mapping
from llm_lifecycle_lab.contracts import (
    SCHEMA_VERSION,
    JsonContract,
    RecordKind,
    utc_now,
)
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.posttraining import (
    PublicPosttrainingRecipe,
    _dedupe_sft_conversations,
    _download_sources,
    _read_source_rows,
    _transform_source,
    _write_jsonl,
)
from llm_lifecycle_lab.data.schemas import (
    format_validation_failure,
    validate_jsonl,
)
from llm_lifecycle_lab.exceptions import ConfigError, DataValidationError


@dataclass(frozen=True, slots=True)
class PublicSFTSourceManifest(JsonContract):
    recipe_id: str
    selection_namespace: str
    records: int
    source_file: str
    source_sha256: str
    license: str
    licenses: tuple[str, ...]
    loader_versions: Mapping[str, str]
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PublicSFTSourceManifest:
        return cls(
            recipe_id=str(value["recipe_id"]),
            selection_namespace=str(value["selection_namespace"]),
            records=int(value["records"]),
            source_file=str(value["source_file"]),
            source_sha256=str(value["source_sha256"]),
            license=str(value["license"]),
            licenses=tuple(str(item) for item in value["licenses"]),
            loader_versions={
                str(name): str(version)
                for name, version in _mapping(
                    value["loader_versions"],
                    "loader_versions",
                ).items()
            },
            created_at=str(value["created_at"]),
            schema_version=str(value["schema_version"]),
        )


def materialize_public_sft(
    recipe_path: str | Path,
    output_dir: str | Path,
    *,
    accepted_licenses: Sequence[str],
) -> PublicSFTSourceManifest:
    """Download pinned public inputs and build only the frozen SFT source."""

    raw = load_mapping(recipe_path)
    recipe = _legacy_transform_recipe(raw)
    recipe_id = _text(raw.get("recipe_id"), "recipe_id")
    namespace = _text(raw.get("selection_namespace"), "selection_namespace")
    output = _mapping(raw.get("output"), "output")
    expected_records = _positive_int(output.get("records"), "output.records")
    expected_sha256 = _text(output.get("sha256"), "output.sha256")

    required = {source.license.casefold(): source.license for source in recipe.sources}
    accepted = {str(value).casefold(): str(value) for value in accepted_licenses}
    if set(accepted) != set(required):
        expected = ", ".join(sorted(required.values()))
        raise DataValidationError(
            f"recipe {recipe_id} requires exactly these --accept-license values: "
            f"{expected}"
        )

    target = Path(output_dir)
    if target.exists():
        raise DataValidationError(
            f"public SFT output already exists; refusing to overwrite: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
    )
    try:
        source_paths, loader_versions = _download_sources(recipe)
        records: list[dict[str, Any]] = []
        for source in recipe.sources:
            rows = _read_source_rows(source_paths[source.source_id], source)
            transformed = _transform_source(rows, recipe, source)
            records.extend(transformed.get(RecordKind.SFT, ()))
        if namespace == "public-60m-v3":
            records = _dedupe_sft_conversations(records)
        records.sort(key=lambda row: row["id"])
        if len(records) != expected_records:
            raise DataValidationError(
                f"public SFT materialization produced {len(records)} records; "
                f"expected {expected_records}"
            )

        source_path = temporary / "source.jsonl"
        _write_jsonl(source_path, records)
        validated = validate_jsonl(source_path, RecordKind.SFT)
        if not validated.report.ok:
            raise DataValidationError(format_validation_failure(validated.report))
        actual_sha256 = sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise DataValidationError(
                "public SFT output hash mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        manifest = PublicSFTSourceManifest(
            recipe_id=recipe_id,
            selection_namespace=namespace,
            records=len(records),
            source_file=source_path.name,
            source_sha256=actual_sha256,
            license=" AND ".join(sorted(required.values())),
            licenses=tuple(dict.fromkeys(source.license for source in recipe.sources)),
            loader_versions=loader_versions,
        )
        (temporary / "sft_source_manifest.json").write_text(
            manifest.to_json() + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_public_sft_source_manifest(
    source_path: str | Path,
) -> PublicSFTSourceManifest | None:
    source = Path(source_path)
    manifest_path = source.parent / "sft_source_manifest.json"
    if source.name != "source.jsonl" or not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise DataValidationError("public SFT source manifest must be an object")
        manifest = PublicSFTSourceManifest.from_dict(value)
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        if isinstance(exc, DataValidationError):
            raise
        raise DataValidationError(
            f"invalid public SFT source manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.source_file != source.name:
        raise DataValidationError(
            "public SFT source manifest points to a different file"
        )
    actual_sha256 = sha256_file(source)
    if actual_sha256 != manifest.source_sha256:
        raise DataValidationError(
            "public SFT source hash mismatch: "
            f"expected {manifest.source_sha256}, got {actual_sha256}"
        )
    return manifest


def _legacy_transform_recipe(raw: Mapping[str, Any]) -> PublicPosttrainingRecipe:
    """Adapt the renamed SFT-only recipe to the proven transform implementation."""

    namespace = _text(raw.get("selection_namespace"), "selection_namespace")
    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ConfigError("SFT recipe sources must be a non-empty list")
    converted_sources = []
    selection: dict[str, Any] = {}
    for raw_source in sources:
        source = dict(_mapping(raw_source, "sources[]"))
        transform = _text(source.get("transform"), "sources[].transform")
        source_id = _text(source.get("source_id"), "sources[].source_id")
        source["stages"] = (
            ["sft", "grpo"] if transform == "msvamp-parallel-integer-v1" else ["sft"]
        )
        converted_sources.append(source)
        raw_selection = dict(_mapping(source.get("selection"), "sources[].selection"))
        raw_selection.pop("reject_synthetic", None)
        raw_selection.pop("reserved_groups", None)
        selection[source_id] = raw_selection

    output = _mapping(raw.get("output"), "output")
    adapted = {
        "schema_version": raw.get("schema_version", SCHEMA_VERSION),
        "recipe_id": namespace,
        "description": raw.get("description"),
        "sources": converted_sources,
        "selection": selection,
        "outputs": {
            "sft": {
                "records": output.get("records"),
                "output_sha256": output.get("sha256"),
            },
            "dpo": {"records": 1, "output_sha256": None},
            "grpo": {"records": 1, "output_sha256": None},
        },
    }
    return PublicPosttrainingRecipe.from_dict(adapted)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"SFT recipe {field} must be a mapping")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"SFT recipe {field} must be non-empty text")
    return value


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"SFT recipe {field} must be a positive integer")
    return value
