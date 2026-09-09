"""Versioned materialization of built-in public pretraining datasets."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from llm_lifecycle_lab.contracts import SCHEMA_VERSION, JsonContract, utc_now
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.exceptions import ConfigError, DataValidationError

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RECIPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")


@dataclass(frozen=True, slots=True)
class PublicDatasetRecipe(JsonContract):
    recipe_id: str
    description: str
    repository: str
    revision: str
    config: str
    split: str
    text_field: str
    source_id_field: str
    license: str
    language: str
    synthetic: bool
    upstream_file: str
    upstream_file_sha256: str
    selection_strategy: str
    max_records: int
    min_characters: int
    expected_source_sha256: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _RECIPE_PATTERN.fullmatch(self.recipe_id):
            raise ConfigError(f"invalid public dataset recipe_id: {self.recipe_id}")
        for name in (
            "description",
            "repository",
            "config",
            "split",
            "text_field",
            "source_id_field",
            "license",
            "language",
        ):
            if not str(getattr(self, name)).strip():
                raise ConfigError(f"public dataset recipe {name} must not be empty")
        if not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ConfigError(
                "public dataset recipe revision must be a 40-character commit SHA"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.upstream_file_sha256):
            raise ConfigError("public dataset upstream file SHA-256 is invalid")
        if self.selection_strategy != "source-prefix":
            raise ConfigError(
                "unsupported public dataset selection strategy: "
                f"{self.selection_strategy}"
            )
        if self.max_records <= 0:
            raise ConfigError("public dataset max_records must be positive")
        if self.min_characters <= 0:
            raise ConfigError("public dataset min_characters must be positive")
        if self.expected_source_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}",
            self.expected_source_sha256,
        ):
            raise ConfigError("public dataset expected output SHA-256 is invalid")
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigError(
                f"unsupported public recipe schema_version: {self.schema_version}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PublicDatasetRecipe:
        try:
            dataset = _require_mapping(value["dataset"], "dataset")
            selection = _require_mapping(value["selection"], "selection")
            if dataset.get("provider") != "huggingface":
                raise ConfigError("only provider=huggingface is currently supported")
            return cls(
                schema_version=str(value["schema_version"]),
                recipe_id=str(value["recipe_id"]),
                description=str(value["description"]),
                repository=str(dataset["repository"]),
                revision=str(dataset["revision"]),
                config=str(dataset["config"]),
                split=str(dataset["split"]),
                text_field=str(dataset["text_field"]),
                source_id_field=str(dataset["source_id_field"]),
                license=str(dataset["license"]),
                language=str(dataset["language"]),
                synthetic=bool(dataset["synthetic"]),
                upstream_file=str(dataset["file"]),
                upstream_file_sha256=str(dataset["file_sha256"]),
                selection_strategy=str(selection["strategy"]),
                max_records=int(selection["max_records"]),
                min_characters=int(selection["min_characters"]),
                expected_source_sha256=(
                    str(selection["output_sha256"])
                    if selection.get("output_sha256") is not None
                    else None
                ),
            )
        except KeyError as exc:
            raise ConfigError(
                f"public dataset recipe is missing field: {exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid public dataset recipe: {exc}") from exc


@dataclass(frozen=True, slots=True)
class PublicSourceManifest(JsonContract):
    recipe_id: str
    repository: str
    revision: str
    config: str
    split: str
    license: str
    language: str
    synthetic: bool
    selection: Mapping[str, Any]
    records: int
    skipped_records: int
    source_file: str
    source_sha256: str
    upstream_file: str
    upstream_file_sha256: str
    loader_versions: Mapping[str, str]
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.records <= 0 or self.skipped_records < 0:
            raise DataValidationError("invalid public source record counts")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise DataValidationError("public source SHA-256 is invalid")
        if not self.upstream_file.strip():
            raise DataValidationError("public upstream file must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.upstream_file_sha256):
            raise DataValidationError("public upstream file SHA-256 is invalid")
        object.__setattr__(
            self,
            "selection",
            MappingProxyType(dict(self.selection)),
        )
        object.__setattr__(
            self,
            "loader_versions",
            MappingProxyType(dict(self.loader_versions)),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise DataValidationError(
                f"unsupported public source schema_version: {self.schema_version}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PublicSourceManifest:
        return cls(
            recipe_id=str(value["recipe_id"]),
            repository=str(value["repository"]),
            revision=str(value["revision"]),
            config=str(value["config"]),
            split=str(value["split"]),
            license=str(value["license"]),
            language=str(value["language"]),
            synthetic=bool(value["synthetic"]),
            selection=_require_mapping(value["selection"], "selection"),
            records=int(value["records"]),
            skipped_records=int(value["skipped_records"]),
            source_file=str(value["source_file"]),
            source_sha256=str(value["source_sha256"]),
            upstream_file=str(value["upstream_file"]),
            upstream_file_sha256=str(value["upstream_file_sha256"]),
            loader_versions={
                str(name): str(version)
                for name, version in _require_mapping(
                    value["loader_versions"],
                    "loader_versions",
                ).items()
            },
            created_at=str(value["created_at"]),
            schema_version=str(value["schema_version"]),
        )


def available_public_recipes() -> tuple[PublicDatasetRecipe, ...]:
    recipe_root = files("llm_lifecycle_lab.data.recipes")
    recipes = [
        _load_recipe_resource(resource)
        for resource in recipe_root.iterdir()
        if resource.name.endswith(".yaml")
    ]
    return tuple(sorted(recipes, key=lambda recipe: recipe.recipe_id))


def load_public_recipe(recipe_id: str) -> PublicDatasetRecipe:
    for recipe in available_public_recipes():
        if recipe.recipe_id == recipe_id:
            return recipe
    available = ", ".join(recipe.recipe_id for recipe in available_public_recipes())
    raise ConfigError(
        f"unknown public dataset recipe {recipe_id!r}; available: {available}"
    )


def materialize_public_dataset(
    recipe_id: str,
    output_dir: str | Path,
    *,
    accepted_license: str,
) -> PublicSourceManifest:
    """Materialize a pinned public dataset into canonical pretrain JSONL."""

    recipe = load_public_recipe(recipe_id)
    if accepted_license.casefold() != recipe.license.casefold():
        raise DataValidationError(
            f"recipe {recipe_id} requires --accept-license {recipe.license}"
        )

    target = Path(output_dir)
    if target.exists():
        raise DataValidationError(
            f"public source output already exists; refusing to overwrite: {target}"
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
            f"cannot create public source directory for {target}: {exc}"
        ) from exc

    source_path = temporary / "source.jsonl"
    try:
        rows, loader_versions = _load_huggingface_rows(recipe)
        records, skipped = _write_canonical_records(
            rows,
            source_path,
            recipe=recipe,
        )
        source_sha256 = sha256_file(source_path)
        if (
            recipe.expected_source_sha256 is not None
            and source_sha256 != recipe.expected_source_sha256
        ):
            raise DataValidationError(
                f"public recipe output hash mismatch: expected "
                f"{recipe.expected_source_sha256}, got {source_sha256}"
            )
        manifest = PublicSourceManifest(
            recipe_id=recipe.recipe_id,
            repository=recipe.repository,
            revision=recipe.revision,
            config=recipe.config,
            split=recipe.split,
            license=recipe.license,
            language=recipe.language,
            synthetic=recipe.synthetic,
            selection={
                "strategy": recipe.selection_strategy,
                "max_records": recipe.max_records,
                "min_characters": recipe.min_characters,
                "expected_source_sha256": recipe.expected_source_sha256,
            },
            records=records,
            skipped_records=skipped,
            source_file=source_path.name,
            source_sha256=source_sha256,
            upstream_file=recipe.upstream_file,
            upstream_file_sha256=recipe.upstream_file_sha256,
            loader_versions=loader_versions,
        )
        _write_json(temporary / "source_manifest.json", manifest.to_dict())
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_public_source_manifest(
    source_path: str | Path,
) -> PublicSourceManifest | None:
    source = Path(source_path)
    manifest_path = source.parent / "source_manifest.json"
    if source.name != "source.jsonl" or not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise DataValidationError("public source manifest must be an object")
        manifest = PublicSourceManifest.from_dict(value)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(
            f"invalid public source manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.source_file != source.name:
        raise DataValidationError(
            "public source manifest points to a different source file"
        )
    verify_public_source_manifest_recipe(manifest)
    actual_sha256 = sha256_file(source)
    if actual_sha256 != manifest.source_sha256:
        raise DataValidationError(
            f"public source hash mismatch: expected {manifest.source_sha256}, "
            f"got {actual_sha256}"
        )
    return manifest


def verify_public_source_manifest_recipe(manifest: PublicSourceManifest) -> None:
    try:
        recipe = load_public_recipe(manifest.recipe_id)
    except ConfigError as exc:
        raise DataValidationError(
            f"public source manifest references an unavailable recipe: "
            f"{manifest.recipe_id}"
        ) from exc

    expected = {
        "repository": recipe.repository,
        "revision": recipe.revision,
        "config": recipe.config,
        "split": recipe.split,
        "license": recipe.license,
        "language": recipe.language,
        "synthetic": recipe.synthetic,
        "records": recipe.max_records,
        "upstream_file": recipe.upstream_file,
        "upstream_file_sha256": recipe.upstream_file_sha256,
        "selection": {
            "strategy": recipe.selection_strategy,
            "max_records": recipe.max_records,
            "min_characters": recipe.min_characters,
        },
    }
    actual = {
        "repository": manifest.repository,
        "revision": manifest.revision,
        "config": manifest.config,
        "split": manifest.split,
        "license": manifest.license,
        "language": manifest.language,
        "synthetic": manifest.synthetic,
        "records": manifest.records,
        "upstream_file": manifest.upstream_file,
        "upstream_file_sha256": manifest.upstream_file_sha256,
        "selection": {
            "strategy": manifest.selection.get("strategy"),
            "max_records": manifest.selection.get("max_records"),
            "min_characters": manifest.selection.get("min_characters"),
        },
    }
    if recipe.expected_source_sha256 is not None:
        expected["source_sha256"] = recipe.expected_source_sha256
        actual["source_sha256"] = manifest.source_sha256
    mismatches = [
        name
        for name, expected_value in expected.items()
        if actual[name] != expected_value
    ]
    if mismatches:
        raise DataValidationError(
            "public source manifest does not match installed recipe "
            f"{manifest.recipe_id}: {', '.join(mismatches)}"
        )


def _load_recipe_resource(resource: Any) -> PublicDatasetRecipe:
    try:
        value = yaml.safe_load(resource.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"cannot load public dataset recipe {resource}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ConfigError(f"public dataset recipe root must be a mapping: {resource}")
    return PublicDatasetRecipe.from_dict(value)


def _load_huggingface_rows(
    recipe: PublicDatasetRecipe,
) -> tuple[Iterable[Mapping[str, Any]], Mapping[str, str]]:
    try:
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DataValidationError(
            "public dataset support is unavailable; run "
            "`python -m pip install -e '.[public-data]'`"
        ) from exc

    try:
        source_path = Path(
            hf_hub_download(
                repo_id=recipe.repository,
                repo_type="dataset",
                filename=recipe.upstream_file,
                revision=recipe.revision,
            )
        )
        actual_sha256 = sha256_file(source_path)
        if actual_sha256 != recipe.upstream_file_sha256:
            raise DataValidationError(
                f"upstream file hash mismatch for {recipe.upstream_file}: "
                f"expected {recipe.upstream_file_sha256}, got {actual_sha256}"
            )
        parquet_file = parquet.ParquetFile(source_path)
        required_columns = {recipe.text_field, recipe.source_id_field}
        missing_columns = required_columns.difference(parquet_file.schema_arrow.names)
        if missing_columns:
            raise DataValidationError(
                f"upstream Parquet is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        versions = {
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "pyarrow": importlib.metadata.version("pyarrow"),
        }
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError(
            f"cannot load public dataset {recipe.repository}@{recipe.revision}: {exc}"
        ) from exc

    def rows() -> Iterable[Mapping[str, Any]]:
        try:
            for batch in parquet_file.iter_batches(
                batch_size=4096,
                columns=[recipe.text_field, recipe.source_id_field],
            ):
                columns = batch.to_pydict()
                for text, source_id in zip(
                    columns[recipe.text_field],
                    columns[recipe.source_id_field],
                    strict=True,
                ):
                    yield {
                        recipe.text_field: text,
                        recipe.source_id_field: source_id,
                    }
        except Exception as exc:
            raise DataValidationError(
                f"cannot read upstream Parquet {recipe.upstream_file}: {exc}"
            ) from exc

    return rows(), versions


def _write_canonical_records(
    rows: Iterable[Mapping[str, Any]],
    path: Path,
    *,
    recipe: PublicDatasetRecipe,
) -> tuple[int, int]:
    records = 0
    skipped = 0
    seen_ids: set[str] = set()
    source_label = f"hf://datasets/{recipe.repository}@{recipe.revision}/{recipe.split}"
    iterator = iter(rows)
    try:
        with path.open("x", encoding="utf-8") as handle:
            for row_index, row in enumerate(iterator):
                text_value = row.get(recipe.text_field)
                if not isinstance(text_value, str):
                    skipped += 1
                    continue
                text = text_value.strip()
                if len(text) < recipe.min_characters:
                    skipped += 1
                    continue
                upstream_id_value = row.get(recipe.source_id_field)
                upstream_id = (
                    str(upstream_id_value).strip()
                    if upstream_id_value is not None
                    else ""
                )
                if not upstream_id:
                    upstream_id = f"row-{row_index}"
                record_id = _record_id(recipe, upstream_id, text)
                if record_id in seen_ids:
                    raise DataValidationError(
                        f"public recipe produced duplicate record ID: {record_id}"
                    )
                seen_ids.add(record_id)
                record = {
                    "id": record_id,
                    "text": text,
                    "source": source_label,
                    "source_id": upstream_id,
                }
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                records += 1
                if records >= recipe.max_records:
                    break
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise DataValidationError(f"cannot write public source {path}: {exc}") from exc
    finally:
        close_iterator = getattr(iterator, "close", None)
        if close_iterator is not None:
            close_iterator()
    if records != recipe.max_records:
        raise DataValidationError(
            f"public dataset ended after {records} accepted records; "
            f"recipe requires {recipe.max_records}"
        )
    return records, skipped


def _record_id(
    recipe: PublicDatasetRecipe,
    upstream_id: str,
    text: str,
) -> str:
    value = "\0".join(
        (
            recipe.repository,
            recipe.revision,
            recipe.split,
            upstream_id,
            text,
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
            f"cannot write public source manifest: {exc}"
        ) from exc


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"public dataset recipe {field_name} must be a mapping")
    return value
