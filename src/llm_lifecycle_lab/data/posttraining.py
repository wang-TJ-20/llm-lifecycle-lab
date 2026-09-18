"""Pinned public-data materialization for SFT, DPO, and GRPO."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from llm_lifecycle_lab.contracts import (
    SCHEMA_VERSION,
    JsonContract,
    RecordKind,
    utc_now,
)
from llm_lifecycle_lab.data.fingerprint import canonical_record_sha256, sha256_file
from llm_lifecycle_lab.data.schemas import format_validation_failure, validate_jsonl
from llm_lifecycle_lab.exceptions import ConfigError, DataValidationError

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_TOKEN_PATTERN = re.compile(r"<\|(?:pad|bos|eos|unk|chat_start|chat_end)\|>")
_STAGES = (RecordKind.SFT, RecordKind.DPO, RecordKind.GRPO)
_TRANSFORM_STAGES = {
    "oasst-ranked-conversation-v1": ("sft",),
    "dolly-single-turn-sft-v1": ("sft",),
    "hc3-human-single-turn-sft-v1": ("sft",),
    "helpsteer3-single-turn-preference-v1": ("dpo",),
    "hh-helpful-single-turn-preference-v1": ("dpo",),
    "cvalues-helpful-single-turn-preference-v1": ("dpo",),
    "msvamp-parallel-integer-v1": ("sft", "grpo"),
}


@dataclass(frozen=True, slots=True)
class PosttrainingSourceRecipe(JsonContract):
    source_id: str
    transform: str
    stages: tuple[str, ...]
    repository: str
    revision: str
    upstream_file: str
    upstream_file_sha256: str
    file_format: str
    license: str
    upstream_split: str

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.source_id):
            raise ConfigError(f"invalid post-training source_id: {self.source_id}")
        if self.transform not in _TRANSFORM_STAGES:
            raise ConfigError(
                f"unsupported post-training transform: {self.transform}"
            )
        if self.stages != _TRANSFORM_STAGES[self.transform]:
            expected = ", ".join(_TRANSFORM_STAGES[self.transform])
            raise ConfigError(
                f"post-training source {self.source_id} stages must be {expected}"
            )
        for name in (
            "repository",
            "upstream_file",
            "license",
            "upstream_split",
        ):
            if not str(getattr(self, name)).strip():
                raise ConfigError(f"post-training source {name} must not be empty")
        if not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ConfigError(
                "post-training source revision must be a 40-character commit SHA"
            )
        if not _SHA256_PATTERN.fullmatch(self.upstream_file_sha256):
            raise ConfigError("post-training upstream file SHA-256 is invalid")
        if self.file_format not in {"parquet", "json", "jsonl", "jsonl.gz"}:
            raise ConfigError(
                f"unsupported post-training source format: {self.file_format}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PosttrainingSourceRecipe:
        try:
            if value.get("provider") != "huggingface":
                raise ConfigError("only provider=huggingface is currently supported")
            return cls(
                source_id=str(value["source_id"]),
                transform=str(value["transform"]),
                stages=tuple(str(item) for item in value["stages"]),
                repository=str(value["repository"]),
                revision=str(value["revision"]),
                upstream_file=str(value["file"]),
                upstream_file_sha256=str(value["file_sha256"]),
                file_format=str(value["format"]),
                license=str(value["license"]),
                upstream_split=str(value["split"]),
            )
        except KeyError as exc:
            raise ConfigError(
                f"post-training source is missing field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True, slots=True)
class PublicPosttrainingRecipe(JsonContract):
    recipe_id: str
    description: str
    sources: tuple[PosttrainingSourceRecipe, ...]
    selection: Mapping[str, Any]
    expected_records: Mapping[str, int]
    expected_source_sha256: Mapping[str, str | None]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.recipe_id):
            raise ConfigError(
                f"invalid public post-training recipe_id: {self.recipe_id}"
            )
        if not self.description.strip():
            raise ConfigError("public post-training description must not be empty")
        source_ids = tuple(source.source_id for source in self.sources)
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise ConfigError("post-training source IDs must be non-empty and unique")
        if set(self.selection) != set(source_ids):
            raise ConfigError(
                "post-training selection keys must exactly match source IDs"
            )
        _validate_selection(self.sources, self.selection)
        expected_keys = {stage.value for stage in _STAGES}
        if (
            set(self.expected_records) != expected_keys
            or set(self.expected_source_sha256) != expected_keys
        ):
            raise ConfigError(
                "post-training outputs must declare sft, dpo, and grpo"
            )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in self.expected_records.values()
        ):
            raise ConfigError(
                "post-training expected record counts must be positive integers"
            )
        for stage, value in self.expected_source_sha256.items():
            if value is not None and not _SHA256_PATTERN.fullmatch(value):
                raise ConfigError(
                    f"post-training {stage} expected output SHA-256 is invalid"
                )
        object.__setattr__(
            self,
            "selection",
            MappingProxyType(_copy_mapping(self.selection)),
        )
        object.__setattr__(
            self,
            "expected_records",
            MappingProxyType(dict(self.expected_records)),
        )
        object.__setattr__(
            self,
            "expected_source_sha256",
            MappingProxyType(dict(self.expected_source_sha256)),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigError(
                f"unsupported post-training recipe schema_version: "
                f"{self.schema_version}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PublicPosttrainingRecipe:
        try:
            raw_sources = value["sources"]
            if not isinstance(raw_sources, list):
                raise ConfigError("post-training sources must be a list")
            outputs = _require_mapping(value["outputs"], "outputs")
            return cls(
                schema_version=str(value["schema_version"]),
                recipe_id=str(value["recipe_id"]),
                description=str(value["description"]),
                sources=tuple(
                    PosttrainingSourceRecipe.from_dict(
                        _require_mapping(item, "sources[]")
                    )
                    for item in raw_sources
                ),
                selection=_require_mapping(value["selection"], "selection"),
                expected_records={
                    stage.value: int(
                        _require_mapping(outputs[stage.value], stage.value)["records"]
                    )
                    for stage in _STAGES
                },
                expected_source_sha256={
                    stage.value: (
                        str(
                            _require_mapping(outputs[stage.value], stage.value)[
                                "output_sha256"
                            ]
                        )
                        if _require_mapping(outputs[stage.value], stage.value).get(
                            "output_sha256"
                        )
                        is not None
                        else None
                    )
                    for stage in _STAGES
                },
            )
        except KeyError as exc:
            raise ConfigError(
                f"public post-training recipe is missing field: {exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid public post-training recipe: {exc}") from exc

    def source(self, source_id: str) -> PosttrainingSourceRecipe:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise ConfigError(f"post-training recipe has no source {source_id!r}")


@dataclass(frozen=True, slots=True)
class PublicPosttrainingSourceManifest(JsonContract):
    manifest_type: str
    recipe_id: str
    record_kind: RecordKind
    license: str
    licenses: tuple[str, ...]
    languages: tuple[str, ...]
    sources: tuple[Mapping[str, Any], ...]
    selection: Mapping[str, Any]
    records: int
    language_records: Mapping[str, int]
    source_file: str
    source_sha256: str
    loader_versions: Mapping[str, str]
    source_records: Mapping[str, int] = field(default_factory=dict)
    task_records: Mapping[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.manifest_type != "public-posttraining-source":
            raise DataValidationError("invalid post-training manifest_type")
        if self.record_kind not in _STAGES:
            raise DataValidationError("invalid post-training record_kind")
        if not self.license.strip() or not self.licenses:
            raise DataValidationError("post-training source licenses must not be empty")
        if self.languages != ("en", "zh"):
            raise DataValidationError(
                "post-training source languages must be en and zh"
            )
        if self.records <= 0 or sum(self.language_records.values()) != self.records:
            raise DataValidationError("invalid post-training source record counts")
        if set(self.language_records) != {"en", "zh"}:
            raise DataValidationError(
                "post-training language records must declare en and zh"
            )
        if any(value <= 0 for value in self.language_records.values()):
            raise DataValidationError(
                "post-training source must contain both en and zh"
            )
        for name, counts in (
            ("source_records", self.source_records),
            ("task_records", self.task_records),
        ):
            if counts and (
                sum(counts.values()) != self.records
                or any(not key or value <= 0 for key, value in counts.items())
            ):
                raise DataValidationError(
                    f"invalid post-training {name} counts"
                )
        if self.source_file != "source.jsonl":
            raise DataValidationError("post-training source_file must be source.jsonl")
        if not _SHA256_PATTERN.fullmatch(self.source_sha256):
            raise DataValidationError("post-training source SHA-256 is invalid")
        object.__setattr__(
            self,
            "sources",
            tuple(MappingProxyType(_copy_mapping(item)) for item in self.sources),
        )
        object.__setattr__(
            self,
            "selection",
            MappingProxyType(_copy_mapping(self.selection)),
        )
        object.__setattr__(
            self,
            "language_records",
            MappingProxyType(dict(self.language_records)),
        )
        object.__setattr__(
            self,
            "loader_versions",
            MappingProxyType(dict(self.loader_versions)),
        )
        object.__setattr__(
            self,
            "source_records",
            MappingProxyType(dict(self.source_records)),
        )
        object.__setattr__(
            self,
            "task_records",
            MappingProxyType(dict(self.task_records)),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise DataValidationError(
                f"unsupported post-training manifest schema_version: "
                f"{self.schema_version}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PublicPosttrainingSourceManifest:
        sources = value["sources"]
        if not isinstance(sources, list):
            raise DataValidationError("post-training manifest sources must be a list")
        return cls(
            manifest_type=str(value["manifest_type"]),
            recipe_id=str(value["recipe_id"]),
            record_kind=RecordKind(value["record_kind"]),
            license=str(value["license"]),
            licenses=tuple(str(item) for item in value["licenses"]),
            languages=tuple(str(item) for item in value["languages"]),
            sources=tuple(
                _require_mapping(item, "manifest.sources[]") for item in sources
            ),
            selection=_require_mapping(value["selection"], "manifest.selection"),
            records=int(value["records"]),
            language_records={
                str(name): int(count)
                for name, count in _require_mapping(
                    value["language_records"],
                    "manifest.language_records",
                ).items()
            },
            source_file=str(value["source_file"]),
            source_sha256=str(value["source_sha256"]),
            loader_versions={
                str(name): str(version)
                for name, version in _require_mapping(
                    value["loader_versions"],
                    "manifest.loader_versions",
                ).items()
            },
            source_records={
                str(name): int(count)
                for name, count in _require_mapping(
                    value.get("source_records", {}),
                    "manifest.source_records",
                ).items()
            },
            task_records={
                str(name): int(count)
                for name, count in _require_mapping(
                    value.get("task_records", {}),
                    "manifest.task_records",
                ).items()
            },
            created_at=str(value["created_at"]),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class PublicPosttrainingBundleManifest(JsonContract):
    manifest_type: str
    recipe_id: str
    stages: tuple[Mapping[str, Any], ...]
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.manifest_type != "public-posttraining-bundle":
            raise DataValidationError("invalid post-training bundle manifest_type")
        if not _ID_PATTERN.fullmatch(self.recipe_id):
            raise DataValidationError("invalid post-training bundle recipe_id")
        if len(self.stages) != len(_STAGES):
            raise DataValidationError(
                "post-training bundle must contain sft, dpo, and grpo"
            )
        stage_names = []
        for item in self.stages:
            if set(item) != {
                "record_kind",
                "manifest",
                "records",
                "source_sha256",
            }:
                raise DataValidationError("invalid post-training bundle stage fields")
            try:
                stage = RecordKind(item["record_kind"])
            except ValueError as exc:
                raise DataValidationError(
                    "invalid post-training bundle record_kind"
                ) from exc
            if stage not in _STAGES:
                raise DataValidationError("invalid post-training bundle record_kind")
            if item["manifest"] != f"{stage.value}/source_manifest.json":
                raise DataValidationError("invalid post-training bundle manifest path")
            records = item["records"]
            if (
                not isinstance(records, int)
                or isinstance(records, bool)
                or records <= 0
            ):
                raise DataValidationError(
                    "post-training bundle records must be positive integers"
                )
            source_sha256 = item["source_sha256"]
            if not isinstance(source_sha256, str) or not _SHA256_PATTERN.fullmatch(
                source_sha256
            ):
                raise DataValidationError(
                    "post-training bundle source SHA-256 is invalid"
                )
            stage_names.append(stage)
        if tuple(stage_names) != _STAGES:
            raise DataValidationError(
                "post-training bundle stages must be ordered as sft, dpo, and grpo"
            )
        object.__setattr__(
            self,
            "stages",
            tuple(MappingProxyType(_copy_mapping(item)) for item in self.stages),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise DataValidationError(
                "unsupported post-training bundle schema_version: "
                f"{self.schema_version}"
            )


def available_public_posttraining_recipes() -> tuple[PublicPosttrainingRecipe, ...]:
    recipe_root = files("llm_lifecycle_lab.data.posttraining_recipes")
    recipes = [
        _load_recipe_resource(resource)
        for resource in recipe_root.iterdir()
        if resource.name.endswith(".yaml")
    ]
    return tuple(sorted(recipes, key=lambda recipe: recipe.recipe_id))


def load_public_posttraining_recipe(recipe_id: str) -> PublicPosttrainingRecipe:
    for recipe in available_public_posttraining_recipes():
        if recipe.recipe_id == recipe_id:
            return recipe
    available = ", ".join(
        recipe.recipe_id for recipe in available_public_posttraining_recipes()
    )
    raise ConfigError(
        f"unknown public post-training recipe {recipe_id!r}; available: {available}"
    )


def materialize_public_posttraining(
    recipe_id: str,
    output_dir: str | Path,
    *,
    accepted_licenses: Sequence[str],
) -> PublicPosttrainingBundleManifest:
    """Materialize one pinned recipe into canonical SFT, DPO, and GRPO sources."""

    recipe = load_public_posttraining_recipe(recipe_id)
    required = {source.license.casefold(): source.license for source in recipe.sources}
    accepted = {value.casefold(): value for value in accepted_licenses}
    if set(accepted) != set(required):
        expected = ", ".join(sorted(required.values()))
        raise DataValidationError(
            f"recipe {recipe_id} requires exactly these --accept-license values: "
            f"{expected}"
        )

    target = Path(output_dir)
    if target.exists():
        raise DataValidationError(
            "public post-training output already exists; refusing to overwrite: "
            f"{target}"
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
        )
    except OSError as exc:
        raise DataValidationError(
            f"cannot create post-training source directory for {target}: {exc}"
        ) from exc

    try:
        records, loader_versions = _load_posttraining_records(recipe)
        stage_manifests = []
        for stage in _STAGES:
            stage_path = temporary / stage.value
            stage_path.mkdir()
            source_path = stage_path / "source.jsonl"
            _write_jsonl(source_path, records[stage])
            validated = validate_jsonl(source_path, stage)
            if not validated.report.ok:
                raise DataValidationError(format_validation_failure(validated.report))
            source_sha256 = sha256_file(source_path)
            expected_sha256 = recipe.expected_source_sha256[stage.value]
            if expected_sha256 is not None and source_sha256 != expected_sha256:
                raise DataValidationError(
                    f"public post-training {stage.value} output hash mismatch: "
                    f"expected {expected_sha256}, got {source_sha256}"
                )
            used_sources = _sources_for_stage(recipe, stage)
            licenses = tuple(dict.fromkeys(source.license for source in used_sources))
            language_records = {
                language: sum(
                    record.get("language") == language for record in records[stage]
                )
                for language in ("en", "zh")
            }
            source_records = {
                source.source_id: sum(
                    record.get("metadata", {}).get("source")
                    == _source_uri(source)
                    for record in records[stage]
                )
                for source in used_sources
            }
            source_records = {
                name: count for name, count in source_records.items() if count
            }
            task_records: dict[str, int] = {}
            for record in records[stage]:
                metadata = record.get("metadata", {})
                task = str(
                    metadata.get("task")
                    or {
                        "oasst-ranked-conversation-v1": "conversation",
                        "helpsteer3-single-turn-preference-v1": (
                            "general-preference"
                        ),
                        "msvamp-parallel-integer-v1": "math-verifiable",
                    }.get(str(metadata.get("transform")), "unspecified")
                )
                task_records[task] = task_records.get(task, 0) + 1
            manifest = PublicPosttrainingSourceManifest(
                manifest_type="public-posttraining-source",
                recipe_id=recipe.recipe_id,
                record_kind=stage,
                license=" AND ".join(sorted(licenses)),
                licenses=licenses,
                languages=("en", "zh"),
                sources=tuple(_source_manifest_value(item) for item in used_sources),
                selection=_selection_for_stage(recipe, stage),
                records=len(records[stage]),
                language_records=language_records,
                source_file=source_path.name,
                source_sha256=source_sha256,
                loader_versions=loader_versions,
                source_records=source_records,
                task_records=task_records,
            )
            _write_json(stage_path / "source_manifest.json", manifest.to_dict())
            stage_manifests.append(manifest)
        bundle = PublicPosttrainingBundleManifest(
            manifest_type="public-posttraining-bundle",
            recipe_id=recipe.recipe_id,
            stages=tuple(
                {
                    "record_kind": manifest.record_kind.value,
                    "manifest": f"{manifest.record_kind.value}/source_manifest.json",
                    "records": manifest.records,
                    "source_sha256": manifest.source_sha256,
                }
                for manifest in stage_manifests
            ),
        )
        _write_json(temporary / "bundle_manifest.json", bundle.to_dict())
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return bundle


def load_public_posttraining_source_manifest(
    source_path: str | Path,
) -> PublicPosttrainingSourceManifest | None:
    source = Path(source_path)
    manifest_path = source.parent / "source_manifest.json"
    if source.name != "source.jsonl" or not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise DataValidationError(
                "public post-training source manifest must be an object"
            )
        if value.get("manifest_type") != "public-posttraining-source":
            return None
        manifest = PublicPosttrainingSourceManifest.from_dict(value)
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
            f"invalid public post-training source manifest {manifest_path}: {exc}"
        ) from exc
    _verify_source_manifest_recipe(manifest)
    actual_sha256 = sha256_file(source)
    if actual_sha256 != manifest.source_sha256:
        raise DataValidationError(
            "public post-training source hash mismatch: "
            f"expected {manifest.source_sha256}, got {actual_sha256}"
        )
    return manifest


def _load_posttraining_records(
    recipe: PublicPosttrainingRecipe,
) -> tuple[
    dict[RecordKind, list[dict[str, Any]]],
    dict[str, str],
]:
    source_paths, loader_versions = _download_sources(recipe)
    records: dict[RecordKind, list[dict[str, Any]]] = {
        stage: [] for stage in _STAGES
    }
    for source in recipe.sources:
        rows = _read_source_rows(source_paths[source.source_id], source)
        transformed = _transform_source(rows, recipe, source)
        for stage, stage_records in transformed.items():
            records[stage].extend(stage_records)
    for stage in _STAGES:
        records[stage].sort(key=lambda row: row["id"])
        expected = recipe.expected_records[stage.value]
        if len(records[stage]) != expected:
            raise DataValidationError(
                f"public post-training {stage.value} produced "
                f"{len(records[stage])} records; expected {expected}"
            )
    return records, loader_versions


def _download_sources(
    recipe: PublicPosttrainingRecipe,
) -> tuple[dict[str, Path], dict[str, str]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DataValidationError(
            "public post-training data support is unavailable; run "
            "`python -m pip install -r requirements.txt`"
        ) from exc

    result = {}
    try:
        for source in recipe.sources:
            path = Path(
                hf_hub_download(
                    repo_id=source.repository,
                    repo_type="dataset",
                    filename=source.upstream_file,
                    revision=source.revision,
                )
            )
            actual_sha256 = sha256_file(path)
            if actual_sha256 != source.upstream_file_sha256:
                raise DataValidationError(
                    f"upstream file hash mismatch for {source.upstream_file}: "
                    f"expected {source.upstream_file_sha256}, got {actual_sha256}"
                )
            result[source.source_id] = path
        versions = {
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "pyarrow": importlib.metadata.version("pyarrow"),
        }
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError(
            f"cannot download public post-training sources: {exc}"
        ) from exc
    return result, versions


def _read_oasst(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise DataValidationError(
            "OASST materialization requires pyarrow; install requirements.txt"
        ) from exc
    columns = (
        "message_id",
        "parent_id",
        "text",
        "role",
        "lang",
        "review_result",
        "deleted",
        "rank",
        "synthetic",
        "message_tree_id",
        "tree_state",
    )
    try:
        table = parquet.read_table(path, columns=list(columns))
        return table.to_pylist()
    except Exception as exc:
        raise DataValidationError(f"cannot read OASST Parquet {path}: {exc}") from exc


def _read_jsonl_gzip(path: Path) -> Iterable[Mapping[str, Any]]:
    def rows() -> Iterable[Mapping[str, Any]]:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                yield from _parse_json_lines(handle, path)
        except OSError as exc:
            raise DataValidationError(
                f"cannot read compressed JSONL {path}: {exc}"
            ) from exc

    return rows()


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    def rows() -> Iterable[Mapping[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                yield from _parse_json_lines(handle, path)
        except OSError as exc:
            raise DataValidationError(f"cannot read JSONL {path}: {exc}") from exc

    return rows()


def _read_json(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise DataValidationError(f"JSON source {path} must contain an object list")
    return value


def _read_source_rows(
    path: Path,
    source: PosttrainingSourceRecipe,
) -> Iterable[Mapping[str, Any]]:
    readers = {
        "parquet": _read_oasst,
        "json": _read_json,
        "jsonl": _read_jsonl,
        "jsonl.gz": _read_jsonl_gzip,
    }
    return readers[source.file_format](path)


def _parse_json_lines(lines: Iterable[str], path: Path) -> Iterable[Mapping[str, Any]]:
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataValidationError(
                f"invalid JSON in {path} at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, Mapping):
            raise DataValidationError(
                f"JSON row in {path} at line {line_number} is not an object"
            )
        yield value


def _transform_oasst(
    rows: Iterable[Mapping[str, Any]],
    recipe: PublicPosttrainingRecipe,
    source: PosttrainingSourceRecipe | None = None,
) -> tuple[list[dict[str, Any]], int]:
    source = source or recipe.source("oasst1")
    config = _require_mapping(
        recipe.selection[source.source_id],
        f"selection.{source.source_id}",
    )
    per_language = int(config["records_per_language"])
    max_messages = int(config["max_messages"])
    max_characters = int(config["max_characters"])
    values = [dict(row) for row in rows]
    by_id: dict[str, dict[str, Any]] = {}
    for row in values:
        message_id = _clean_text(row.get("message_id"))
        if message_id is None:
            continue
        if message_id in by_id:
            raise DataValidationError(f"OASST duplicate message_id: {message_id}")
        by_id[message_id] = row

    candidates: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "en": [],
        "zh": [],
    }
    skipped = 0
    for row in values:
        language = row.get("lang")
        if (
            language not in candidates
            or row.get("role") != "assistant"
            or row.get("rank") != 0
        ):
            skipped += 1
            continue
        chain = _oasst_chain(row, by_id)
        if chain is None or not _valid_oasst_chain(
            chain,
            language=language,
            max_messages=max_messages,
            max_characters=max_characters,
        ):
            skipped += 1
            continue
        message_id = str(row["message_id"])
        tree_id = str(row["message_tree_id"])
        messages = [
            {
                "role": "user" if item["role"] == "prompter" else "assistant",
                "content": str(item["text"]).strip(),
            }
            for item in chain
        ]
        record = {
            "id": _record_id(source, "sft", message_id, language),
            "source_id": f"{source.source_id}:{tree_id}",
            "language": language,
            "messages": messages,
            "metadata": {
                "source": _source_uri(source),
                "transform": source.transform,
                "upstream_message_id": message_id,
            },
        }
        key = _selection_key(
            recipe.recipe_id, source.source_id, language, message_id
        )
        candidates[language].append((key, record))

    selected = []
    for language in ("en", "zh"):
        ordered = sorted(
            candidates[language], key=lambda item: (item[0], item[1]["id"])
        )
        if len(ordered) < per_language:
            raise DataValidationError(
                f"OASST has only {len(ordered)} eligible {language} records; "
                f"recipe requires {per_language}"
            )
        selected.extend(record for _, record in ordered[:per_language])
    return selected, skipped


def _oasst_chain(
    leaf: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]] | None:
    chain = []
    current: Mapping[str, Any] | None = leaf
    seen = set()
    while current is not None:
        message_id = _clean_text(current.get("message_id"))
        if message_id is None or message_id in seen:
            return None
        seen.add(message_id)
        chain.append(current)
        parent_id = _clean_text(current.get("parent_id"))
        if parent_id is None:
            current = None
        else:
            current = by_id.get(parent_id)
            if current is None:
                return None
    chain.reverse()
    return chain


def _valid_oasst_chain(
    chain: Sequence[Mapping[str, Any]],
    *,
    language: str,
    max_messages: int,
    max_characters: int,
) -> bool:
    if not chain or len(chain) > max_messages or len(chain) % 2 != 0:
        return False
    character_count = 0
    for index, row in enumerate(chain):
        expected_role = "prompter" if index % 2 == 0 else "assistant"
        text = _clean_text(row.get("text"))
        if (
            row.get("role") != expected_role
            or row.get("lang") != language
            or row.get("review_result") is not True
            or row.get("deleted") is not False
            or row.get("synthetic") is not False
            or row.get("tree_state") != "ready_for_export"
            or text is None
            or _CONTROL_TOKEN_PATTERN.search(text)
            or (expected_role == "assistant" and row.get("rank") != 0)
        ):
            return False
        character_count += len(text)
    return character_count <= max_characters


def _transform_helpsteer(
    rows: Iterable[Mapping[str, Any]],
    recipe: PublicPosttrainingRecipe,
    source: PosttrainingSourceRecipe | None = None,
) -> tuple[list[dict[str, Any]], int]:
    source = source or recipe.source("helpsteer3")
    config = _require_mapping(
        recipe.selection[source.source_id],
        f"selection.{source.source_id}",
    )
    per_language = int(config["records_per_language"])
    max_characters = int(config["max_characters"])
    context_policy = config["context_messages"]
    language_map = {"english": "en", "chinese": "zh"}
    candidates: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "en": [],
        "zh": [],
    }
    seen_pairs = set()
    skipped = 0
    for row in rows:
        language = language_map.get(str(row.get("language", "")).casefold())
        context = row.get("context")
        response1 = _clean_text(row.get("response1"))
        response2 = _clean_text(row.get("response2"))
        preference = row.get("overall_preference")
        if (
            language is None
            or not isinstance(context, list)
            or not context
            or (context_policy == 1 and len(context) != 1)
            or not isinstance(context[-1], Mapping)
            or context[-1].get("role") != "user"
            or not isinstance(preference, (int, float))
            or isinstance(preference, bool)
            or preference == 0
            or response1 is None
            or response2 is None
        ):
            skipped += 1
            continue
        prompt = _clean_text(context[-1].get("content"))
        if (
            prompt is None
            or _CONTROL_TOKEN_PATTERN.search(prompt)
            or _CONTROL_TOKEN_PATTERN.search(response1)
            or _CONTROL_TOKEN_PATTERN.search(response2)
            or response1 == response2
            or len(prompt) + max(len(response1), len(response2)) > max_characters
        ):
            skipped += 1
            continue
        chosen, rejected = (
            (response1, response2) if preference < 0 else (response2, response1)
        )
        identity = canonical_record_sha256(
            {
                "prompt": _normalize(prompt),
                "chosen": _normalize(chosen),
                "rejected": _normalize(rejected),
            }
        )
        if identity in seen_pairs:
            skipped += 1
            continue
        seen_pairs.add(identity)
        upstream_id = f"{source.source_id}:{identity}"
        record = {
            "id": _record_id(source, "dpo", identity, language),
            "source_id": upstream_id,
            "language": language,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "source": _source_uri(source),
                "transform": source.transform,
                "domain": str(row.get("domain", "")),
                "preference_strength": abs(float(preference)),
            },
        }
        key = _selection_key(
            recipe.recipe_id, source.source_id, language, identity
        )
        candidates[language].append((key, record))

    selected = []
    for language in ("en", "zh"):
        ordered = sorted(
            candidates[language], key=lambda item: (item[0], item[1]["id"])
        )
        if len(ordered) < per_language:
            raise DataValidationError(
                f"HelpSteer3 has only {len(ordered)} eligible {language} records; "
                f"recipe requires {per_language}"
            )
        selected.extend(record for _, record in ordered[:per_language])
    return selected, skipped


def _transform_dolly(
    rows: Iterable[Mapping[str, Any]],
    recipe: PublicPosttrainingRecipe,
    source: PosttrainingSourceRecipe,
) -> tuple[list[dict[str, Any]], int]:
    config = _require_mapping(
        recipe.selection[source.source_id],
        f"selection.{source.source_id}",
    )
    required = int(config["records"])
    max_characters = int(config["max_characters"])
    candidates: list[tuple[str, dict[str, Any]]] = []
    seen = set()
    skipped = 0
    for row in rows:
        instruction = _clean_text(row.get("instruction"))
        context = _clean_text(row.get("context"))
        response = _clean_text(row.get("response"))
        if instruction is None or response is None:
            skipped += 1
            continue
        prompt = (
            f"{instruction}\n\nContext:\n{context}"
            if context is not None
            else instruction
        )
        if (
            _has_control_token(prompt, response)
            or len(prompt) + len(response) > max_characters
        ):
            skipped += 1
            continue
        identity = canonical_record_sha256(
            {"prompt": _normalize(prompt), "response": _normalize(response)}
        )
        if identity in seen:
            skipped += 1
            continue
        seen.add(identity)
        record = {
            "id": _record_id(source, "sft", identity, "en"),
            "source_id": f"{source.source_id}:{identity}",
            "language": "en",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "source": _source_uri(source),
                "transform": source.transform,
                "task": str(row.get("category", "instruction")),
            },
        }
        candidates.append(
            (
                _selection_key(recipe.recipe_id, source.source_id, identity),
                record,
            )
        )
    return _select_candidates(candidates, required, source.source_id), skipped


def _transform_hc3(
    rows: Iterable[Mapping[str, Any]],
    recipe: PublicPosttrainingRecipe,
    source: PosttrainingSourceRecipe,
) -> tuple[list[dict[str, Any]], int]:
    config = _require_mapping(
        recipe.selection[source.source_id],
        f"selection.{source.source_id}",
    )
    required = int(config["records"])
    max_characters = int(config["max_characters"])
    candidates: list[tuple[str, dict[str, Any]]] = []
    seen = set()
    skipped = 0
    for row in rows:
        prompt = _clean_text(row.get("question"))
        answers = row.get("human_answers")
        response = (
            _clean_text(answers[0])
            if isinstance(answers, list) and answers
            else None
        )
        if (
            prompt is None
            or response is None
            or _has_control_token(prompt, response)
            or len(prompt) + len(response) > max_characters
        ):
            skipped += 1
            continue
        identity = canonical_record_sha256(
            {"prompt": _normalize(prompt), "response": _normalize(response)}
        )
        if identity in seen:
            skipped += 1
            continue
        seen.add(identity)
        record = {
            "id": _record_id(source, "sft", identity, "zh"),
            "source_id": f"{source.source_id}:{identity}",
            "language": "zh",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "source": _source_uri(source),
                "transform": source.transform,
                "task": str(row.get("source", "open_qa")),
                "answer_origin": "human",
            },
        }
        candidates.append(
            (
                _selection_key(recipe.recipe_id, source.source_id, identity),
                record,
            )
        )
    return _select_candidates(candidates, required, source.source_id), skipped


def _transform_hh_helpful(
    rows: Iterable[Mapping[str, Any]],
    recipe: PublicPosttrainingRecipe,
    source: PosttrainingSourceRecipe,
) -> tuple[list[dict[str, Any]], int]:
    config = _require_mapping(
        recipe.selection[source.source_id],
        f"selection.{source.source_id}",
    )
    required = int(config["records"])
    max_characters = int(config["max_characters"])
    candidates: list[tuple[str, dict[str, Any]]] = []
    seen = set()
    skipped = 0
    for row in rows:
        chosen = _final_hh_turn(row.get("chosen"))
        rejected = _final_hh_turn(row.get("rejected"))
        if chosen is None or rejected is None:
            skipped += 1
            continue
        chosen_prompt, chosen_response = chosen
        rejected_prompt, rejected_response = rejected
        if (
            _normalize(chosen_prompt) != _normalize(rejected_prompt)
            or chosen_response == rejected_response
            or _has_control_token(
                chosen_prompt, chosen_response, rejected_response
            )
            or len(chosen_prompt)
            + max(len(chosen_response), len(rejected_response))
            > max_characters
        ):
            skipped += 1
            continue
        identity = canonical_record_sha256(
            {
                "prompt": _normalize(chosen_prompt),
                "chosen": _normalize(chosen_response),
                "rejected": _normalize(rejected_response),
            }
        )
        if identity in seen:
            skipped += 1
            continue
        seen.add(identity)
        record = {
            "id": _record_id(source, "dpo", identity, "en"),
            "source_id": f"{source.source_id}:{identity}",
            "language": "en",
            "prompt": chosen_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "metadata": {
                "source": _source_uri(source),
                "transform": source.transform,
                "task": "helpfulness",
            },
        }
        candidates.append(
            (
                _selection_key(recipe.recipe_id, source.source_id, identity),
                record,
            )
        )
    return _select_candidates(candidates, required, source.source_id), skipped


def _transform_cvalues(
    rows: Iterable[Mapping[str, Any]],
    recipe: PublicPosttrainingRecipe,
    source: PosttrainingSourceRecipe,
) -> tuple[list[dict[str, Any]], int]:
    config = _require_mapping(
        recipe.selection[source.source_id],
        f"selection.{source.source_id}",
    )
    required = int(config["records"])
    max_characters = int(config["max_characters"])
    candidates: list[tuple[str, dict[str, Any]]] = []
    seen = set()
    skipped = 0
    for row in rows:
        prompt = _clean_text(row.get("prompt"))
        chosen = _clean_text(row.get("pos_resp"))
        rejected = _clean_text(row.get("neg_resp"))
        if (
            prompt is None
            or chosen is None
            or rejected is None
            or chosen == rejected
            or _has_control_token(prompt, chosen, rejected)
            or len(prompt) + max(len(chosen), len(rejected)) > max_characters
        ):
            skipped += 1
            continue
        identity = canonical_record_sha256(
            {
                "prompt": _normalize(prompt),
                "chosen": _normalize(chosen),
                "rejected": _normalize(rejected),
            }
        )
        if identity in seen:
            skipped += 1
            continue
        seen.add(identity)
        record = {
            "id": _record_id(source, "dpo", identity, "zh"),
            "source_id": f"{source.source_id}:{identity}",
            "language": "zh",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "source": _source_uri(source),
                "transform": source.transform,
                "task": "values-alignment",
                "chosen_type": str(row.get("pos_type", "")),
                "rejected_type": str(row.get("neg_type", "")),
            },
        }
        candidates.append(
            (
                _selection_key(recipe.recipe_id, source.source_id, identity),
                record,
            )
        )
    return _select_candidates(candidates, required, source.source_id), skipped


def _transform_msvamp(
    rows: Iterable[Mapping[str, Any]],
    recipe: PublicPosttrainingRecipe,
    source: PosttrainingSourceRecipe | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = source or recipe.source("msvamp")
    config = _require_mapping(
        recipe.selection[source.source_id],
        f"selection.{source.source_id}",
    )
    warmup_groups = int(config["sft_warmup_groups"])
    expected_groups = int(config["expected_groups"])
    groups = []
    seen = set()
    for row in rows:
        query = _clean_text(row.get("query"))
        chinese_query = _clean_text(row.get("m_query"))
        equation = _clean_text(row.get("equation"))
        answer = _integer_answer(row.get("response"))
        if (
            query is None
            or chinese_query is None
            or equation is None
            or answer is None
            or _CONTROL_TOKEN_PATTERN.search(query)
            or _CONTROL_TOKEN_PATTERN.search(chinese_query)
        ):
            raise DataValidationError("MSVAMP contains an invalid training row")
        identity = canonical_record_sha256(
            {
                "query": _normalize(query),
                "m_query": _normalize(chinese_query),
                "equation": equation,
                "answer": answer,
            }
        )
        if identity in seen:
            raise DataValidationError(f"MSVAMP duplicate source group: {identity}")
        seen.add(identity)
        groups.append(
            (
                _selection_key(recipe.recipe_id, source.source_id, identity),
                identity,
                query,
                chinese_query,
                equation,
                answer,
            )
        )
    if len(groups) != expected_groups:
        raise DataValidationError(
            f"MSVAMP has {len(groups)} source groups; expected {expected_groups}"
        )
    groups.sort(key=lambda item: (item[0], item[1]))
    sft = []
    grpo = []
    for index, (_, identity, query, chinese_query, equation, answer) in enumerate(
        groups
    ):
        partition = "sft-warmup" if index < warmup_groups else "grpo"
        source_id = f"{source.source_id}:{identity}"
        for language, prompt in (("en", query), ("zh", chinese_query)):
            common = {
                "id": _record_id(source, partition, identity, language),
                "source_id": source_id,
                "language": language,
            }
            metadata = {
                "source": _source_uri(source),
                "upstream_split": source.upstream_split,
                "intended_use": "training",
                "partition": partition,
                "equation": equation,
                "transform": source.transform,
            }
            if partition == "sft-warmup":
                sft.append(
                    {
                        **common,
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer},
                        ],
                        "metadata": metadata,
                    }
                )
            else:
                grpo.append(
                    {
                        **common,
                        "prompt": prompt,
                        "answer": answer,
                        "metadata": {**metadata, "verifier": "integer"},
                    }
                )
    return sft, grpo


def _transform_source(
    rows: Iterable[Mapping[str, Any]],
    recipe: PublicPosttrainingRecipe,
    source: PosttrainingSourceRecipe,
) -> dict[RecordKind, list[dict[str, Any]]]:
    if source.transform == "oasst-ranked-conversation-v1":
        records, _ = _transform_oasst(rows, recipe, source)
        return {RecordKind.SFT: records}
    if source.transform == "dolly-single-turn-sft-v1":
        records, _ = _transform_dolly(rows, recipe, source)
        return {RecordKind.SFT: records}
    if source.transform == "hc3-human-single-turn-sft-v1":
        records, _ = _transform_hc3(rows, recipe, source)
        return {RecordKind.SFT: records}
    if source.transform == "helpsteer3-single-turn-preference-v1":
        records, _ = _transform_helpsteer(rows, recipe, source)
        return {RecordKind.DPO: records}
    if source.transform == "hh-helpful-single-turn-preference-v1":
        records, _ = _transform_hh_helpful(rows, recipe, source)
        return {RecordKind.DPO: records}
    if source.transform == "cvalues-helpful-single-turn-preference-v1":
        records, _ = _transform_cvalues(rows, recipe, source)
        return {RecordKind.DPO: records}
    if source.transform == "msvamp-parallel-integer-v1":
        sft, grpo = _transform_msvamp(rows, recipe, source)
        return {RecordKind.SFT: sft, RecordKind.GRPO: grpo}
    raise DataValidationError(
        f"no implementation for post-training transform {source.transform}"
    )


def _sources_for_stage(
    recipe: PublicPosttrainingRecipe,
    stage: RecordKind,
) -> tuple[PosttrainingSourceRecipe, ...]:
    return tuple(source for source in recipe.sources if stage.value in source.stages)


def _selection_for_stage(
    recipe: PublicPosttrainingRecipe,
    stage: RecordKind,
) -> dict[str, Any]:
    sources = _sources_for_stage(recipe, stage)
    result = {
        source.source_id: _copy_mapping(recipe.selection[source.source_id])
        for source in sources
    }
    for source in sources:
        if source.transform == "msvamp-parallel-integer-v1":
            result[source.source_id]["partition"] = (
                "lowest-stable-hash-rank"
                if stage is RecordKind.SFT
                else "remaining-stable-hash-ranks"
            )
    return result


def _verify_source_manifest_recipe(
    manifest: PublicPosttrainingSourceManifest,
) -> None:
    try:
        recipe = load_public_posttraining_recipe(manifest.recipe_id)
    except ConfigError as exc:
        raise DataValidationError(
            "public post-training source manifest references an unavailable recipe: "
            f"{manifest.recipe_id}"
        ) from exc
    sources = _sources_for_stage(recipe, manifest.record_kind)
    licenses = tuple(dict.fromkeys(source.license for source in sources))
    expected_records = recipe.expected_records[manifest.record_kind.value]
    expected = {
        "license": " AND ".join(sorted(licenses)),
        "licenses": licenses,
        "sources": tuple(_source_manifest_value(item) for item in sources),
        "selection": _selection_for_stage(recipe, manifest.record_kind),
        "records": expected_records,
    }
    actual = {
        "license": manifest.license,
        "licenses": manifest.licenses,
        "sources": tuple(dict(item) for item in manifest.sources),
        "selection": _copy_mapping(manifest.selection),
        "records": manifest.records,
    }
    expected_sha256 = recipe.expected_source_sha256[manifest.record_kind.value]
    if expected_sha256 is not None:
        expected["source_sha256"] = expected_sha256
        actual["source_sha256"] = manifest.source_sha256
    mismatches = [
        name
        for name, expected_value in expected.items()
        if actual[name] != expected_value
    ]
    if mismatches:
        raise DataValidationError(
            "public post-training source manifest does not match installed recipe "
            f"{manifest.recipe_id}: {', '.join(mismatches)}"
        )


def _validate_selection(
    sources: Sequence[PosttrainingSourceRecipe],
    selection: Mapping[str, Any],
) -> None:
    by_transform = {
        source.transform: (source, _require_mapping(
            selection[source.source_id],
            f"selection.{source.source_id}",
        ))
        for source in sources
    }
    if len(by_transform) != len(sources):
        raise ConfigError("post-training transforms may appear only once per recipe")
    for source, config in by_transform.values():
        if source.transform in {
            "dolly-single-turn-sft-v1",
            "hc3-human-single-turn-sft-v1",
            "hh-helpful-single-turn-preference-v1",
            "cvalues-helpful-single-turn-preference-v1",
        }:
            if set(config) != {"strategy", "records", "max_characters"}:
                raise ConfigError(
                    f"invalid {source.source_id} selection fields"
                )
            if config["strategy"] != "stable-hash-rank":
                raise ConfigError(
                    f"unsupported {source.source_id} selection strategy"
                )
            _require_positive_integers(
                config, ("records", "max_characters"), source.source_id
            )
            continue
        if source.transform == "oasst-ranked-conversation-v1":
            _validate_oasst_selection(config)
        elif source.transform == "helpsteer3-single-turn-preference-v1":
            _validate_helpsteer_selection(config)
        elif source.transform == "msvamp-parallel-integer-v1":
            _validate_msvamp_selection(config)


def _validate_oasst_selection(oasst: Mapping[str, Any]) -> None:
    if set(oasst) != {
        "strategy",
        "records_per_language",
        "max_messages",
        "max_characters",
        "require_all_assistant_rank_zero",
    }:
        raise ConfigError("invalid OASST selection fields")
    if oasst["strategy"] != "stable-hash-rank":
        raise ConfigError("unsupported OASST selection strategy")
    if oasst["require_all_assistant_rank_zero"] is not True:
        raise ConfigError("OASST selection must require all assistant rank zero")

    _require_positive_integers(
        oasst,
        ("records_per_language", "max_messages", "max_characters"),
        "OASST",
    )
    if int(oasst["max_messages"]) % 2 != 0:
        raise ConfigError("OASST max_messages must be even")


def _validate_helpsteer_selection(helpsteer: Mapping[str, Any]) -> None:
    if set(helpsteer) != {
        "strategy",
        "records_per_language",
        "max_characters",
        "context_messages",
        "nonzero_preference",
    }:
        raise ConfigError("invalid HelpSteer3 selection fields")
    if (
        helpsteer["strategy"] != "stable-hash-rank"
        or helpsteer["context_messages"] not in {1, "last-user"}
        or helpsteer["nonzero_preference"] is not True
    ):
        raise ConfigError("invalid HelpSteer3 selection policy")

    _require_positive_integers(
        helpsteer,
        ("records_per_language", "max_characters"),
        "HelpSteer3",
    )


def _validate_msvamp_selection(msvamp: Mapping[str, Any]) -> None:
    if set(msvamp) != {
        "strategy",
        "expected_groups",
        "sft_warmup_groups",
    }:
        raise ConfigError("invalid MSVAMP selection fields")
    if msvamp["strategy"] != "stable-hash-partition":
        raise ConfigError("unsupported MSVAMP selection strategy")

    _require_positive_integers(
        msvamp,
        ("expected_groups", "sft_warmup_groups"),
        "MSVAMP",
    )
    if int(msvamp["sft_warmup_groups"]) >= int(msvamp["expected_groups"]):
        raise ConfigError("MSVAMP warmup must leave at least one GRPO group")


def _require_positive_integers(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    source_name: str,
) -> None:
    if any(
        not isinstance(mapping[name], int)
        or isinstance(mapping[name], bool)
        or mapping[name] <= 0
        for name in names
    ):
        raise ConfigError(
            f"{source_name} selection counts must be positive integers"
        )


def _load_recipe_resource(resource: Any) -> PublicPosttrainingRecipe:
    try:
        value = yaml.safe_load(resource.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"cannot load public post-training recipe {resource}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ConfigError(
            f"public post-training recipe root must be a mapping: {resource}"
        )
    return PublicPosttrainingRecipe.from_dict(value)


def _source_manifest_value(source: PosttrainingSourceRecipe) -> dict[str, Any]:
    return {
        "provider": "huggingface",
        "source_id": source.source_id,
        "repository": source.repository,
        "revision": source.revision,
        "file": source.upstream_file,
        "file_sha256": source.upstream_file_sha256,
        "format": source.file_format,
        "license": source.license,
        "split": source.upstream_split,
    }


def _source_uri(source: PosttrainingSourceRecipe) -> str:
    return f"hf://datasets/{source.repository}@{source.revision}/{source.upstream_file}"


def _record_id(
    source: PosttrainingSourceRecipe,
    transform: str,
    upstream_id: str,
    language: str,
) -> str:
    value = "\0".join(
        (source.repository, source.revision, transform, upstream_id, language)
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _selection_key(recipe_id: str, *parts: str) -> str:
    value = "\0".join((recipe_id, *parts))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _select_candidates(
    candidates: Sequence[tuple[str, dict[str, Any]]],
    required: int,
    source_id: str,
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: (item[0], item[1]["id"]))
    if len(ordered) < required:
        raise DataValidationError(
            f"{source_id} has only {len(ordered)} eligible records; "
            f"recipe requires {required}"
        )
    return [record for _, record in ordered[:required]]


def _final_hh_turn(value: Any) -> tuple[str, str] | None:
    text = _clean_text(value)
    if text is None:
        return None
    _, marker, tail = text.rpartition("\n\nHuman:")
    if not marker or "\n\nAssistant:" not in tail:
        return None
    prompt, separator, response = tail.partition("\n\nAssistant:")
    clean_prompt = _clean_text(prompt)
    clean_response = _clean_text(response)
    if (
        not separator
        or clean_prompt is None
        or clean_response is None
        or "\n\nHuman:" in clean_response
        or "\n\nAssistant:" in clean_response
    ):
        return None
    return clean_prompt, clean_response


def _has_control_token(*values: str) -> bool:
    return any(_CONTROL_TOKEN_PATTERN.search(value) is not None for value in values)


def _integer_answer(value: Any) -> str | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    return str(int(number))


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    return result or None


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _copy_mapping(item) if isinstance(item, Mapping) else item
        for key, item in value.items()
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, TypeError, ValueError) as exc:
        raise DataValidationError(
            f"cannot write public post-training source {path}: {exc}"
        ) from exc


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise DataValidationError(
            f"cannot write public post-training manifest: {exc}"
        ) from exc


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"post-training recipe {field_name} must be a mapping")
    return value
