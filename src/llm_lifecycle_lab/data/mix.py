"""Deterministic composition of pinned public dataset sources."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from llm_lifecycle_lab.contracts import SCHEMA_VERSION, JsonContract, utc_now
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.public import (
    PublicSourceManifest,
    load_public_source_manifest,
    verify_public_source_manifest_recipe,
)
from llm_lifecycle_lab.exceptions import ConfigError, DataValidationError

_MIXTURE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PublicMixtureRecipe(JsonContract):
    mixture_id: str
    description: str
    components: tuple[str, ...]
    strategy: str
    annotate_component: bool = False
    expected_source_sha256: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _MIXTURE_ID_PATTERN.fullmatch(self.mixture_id):
            raise ConfigError(f"invalid public mixture_id: {self.mixture_id}")
        if not self.description.strip():
            raise ConfigError("public mixture description must not be empty")
        if len(self.components) < 2:
            raise ConfigError("public mixture requires at least two components")
        if len(self.components) != len(set(self.components)):
            raise ConfigError("public mixture components must be unique")
        if self.strategy != "round-robin":
            raise ConfigError(f"unsupported public mixture strategy: {self.strategy}")
        if self.expected_source_sha256 is not None and not _SHA256_PATTERN.fullmatch(
            self.expected_source_sha256
        ):
            raise ConfigError("public mixture expected output SHA-256 is invalid")
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigError(
                f"unsupported public mixture schema_version: {self.schema_version}"
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PublicMixtureRecipe:
        try:
            mixing = value["mixing"]
            if not isinstance(mixing, dict):
                raise ConfigError("public mixture mixing must be a mapping")
            components = value["components"]
            if not isinstance(components, list):
                raise ConfigError("public mixture components must be a list")
            return cls(
                schema_version=str(value["schema_version"]),
                mixture_id=str(value["mixture_id"]),
                description=str(value["description"]),
                components=tuple(str(item) for item in components),
                strategy=str(mixing["strategy"]),
                annotate_component=bool(mixing.get("annotate_component", False)),
                expected_source_sha256=(
                    str(mixing["output_sha256"])
                    if mixing.get("output_sha256") is not None
                    else None
                ),
            )
        except KeyError as exc:
            raise ConfigError(
                f"public mixture recipe is missing field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True, slots=True)
class PublicMixtureManifest(JsonContract):
    mixture_id: str
    strategy: str
    components: tuple[dict[str, Any], ...]
    records: int
    source_file: str
    source_sha256: str
    license: str
    languages: tuple[str, ...]
    annotate_component: bool = False
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _MIXTURE_ID_PATTERN.fullmatch(self.mixture_id):
            raise DataValidationError(f"invalid public mixture_id: {self.mixture_id}")
        if self.strategy != "round-robin":
            raise DataValidationError(
                f"unsupported public mixture strategy: {self.strategy}"
            )
        if len(self.components) < 2:
            raise DataValidationError("public mixture requires at least two components")
        if self.records <= 0:
            raise DataValidationError("public mixture records must be positive")
        if self.source_file != "source.jsonl":
            raise DataValidationError("public mixture source_file must be source.jsonl")
        if not _SHA256_PATTERN.fullmatch(self.source_sha256):
            raise DataValidationError("public mixture source SHA-256 is invalid")
        if not self.license.strip():
            raise DataValidationError("public mixture license must not be empty")
        if not self.languages:
            raise DataValidationError("public mixture languages must not be empty")
        object.__setattr__(
            self,
            "components",
            tuple(MappingProxyType(dict(item)) for item in self.components),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise DataValidationError(
                f"unsupported public mixture schema_version: {self.schema_version}"
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PublicMixtureManifest:
        components = value["components"]
        if not isinstance(components, list) or not all(
            isinstance(item, dict) for item in components
        ):
            raise DataValidationError(
                "public mixture manifest components must be objects"
            )
        languages = value["languages"]
        if not isinstance(languages, list):
            raise DataValidationError(
                "public mixture manifest languages must be a list"
            )
        return cls(
            mixture_id=str(value["mixture_id"]),
            strategy=str(value["strategy"]),
            components=tuple(dict(item) for item in components),
            records=int(value["records"]),
            source_file=str(value["source_file"]),
            source_sha256=str(value["source_sha256"]),
            license=str(value["license"]),
            languages=tuple(str(item) for item in languages),
            annotate_component=bool(value.get("annotate_component", False)),
            created_at=str(value["created_at"]),
            schema_version=str(value["schema_version"]),
        )


def available_public_mixture_recipes() -> tuple[PublicMixtureRecipe, ...]:
    recipe_root = files("llm_lifecycle_lab.data.mixtures")
    recipes = [
        _load_mixture_recipe_resource(resource)
        for resource in recipe_root.iterdir()
        if resource.name.endswith(".yaml")
    ]
    return tuple(sorted(recipes, key=lambda recipe: recipe.mixture_id))


def load_public_mixture_recipe(mixture_id: str) -> PublicMixtureRecipe:
    for recipe in available_public_mixture_recipes():
        if recipe.mixture_id == mixture_id:
            return recipe
    available = ", ".join(
        recipe.mixture_id for recipe in available_public_mixture_recipes()
    )
    raise ConfigError(
        f"unknown public mixture recipe {mixture_id!r}; available: {available}"
    )


def materialize_public_mixture(
    mixture_id: str,
    source_paths: list[str | Path],
    output_dir: str | Path,
) -> PublicMixtureManifest:
    recipe = load_public_mixture_recipe(mixture_id)
    components = _load_components(source_paths)
    expected_ids = set(recipe.components)
    actual_ids = set(components)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise DataValidationError(
            f"public mixture inputs do not match {mixture_id}: {'; '.join(details)}"
        )

    ordered = [components[recipe_id] for recipe_id in recipe.components]
    target = Path(output_dir)
    if target.exists():
        raise DataValidationError(
            f"public mixture output already exists; refusing to overwrite: {target}"
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
            f"cannot create public mixture directory for {target}: {exc}"
        ) from exc

    source_path = temporary / "source.jsonl"
    try:
        records = _write_round_robin(
            ordered,
            source_path,
            annotate_component=recipe.annotate_component,
        )
        source_sha256 = sha256_file(source_path)
        if (
            recipe.expected_source_sha256 is not None
            and source_sha256 != recipe.expected_source_sha256
        ):
            raise DataValidationError(
                f"public mixture output hash mismatch: expected "
                f"{recipe.expected_source_sha256}, got {source_sha256}"
            )
        source_manifests = [manifest for _, manifest in ordered]
        manifest = PublicMixtureManifest(
            mixture_id=recipe.mixture_id,
            strategy=recipe.strategy,
            components=tuple(
                source_manifest.to_dict() for source_manifest in source_manifests
            ),
            records=records,
            source_file=source_path.name,
            source_sha256=source_sha256,
            license=_license_expression(source_manifests),
            languages=_languages(source_manifests),
            annotate_component=recipe.annotate_component,
        )
        _write_json(temporary / "mixture_manifest.json", manifest.to_dict())
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_public_mixture_manifest(
    source_path: str | Path,
) -> PublicMixtureManifest | None:
    source = Path(source_path)
    manifest_path = source.parent / "mixture_manifest.json"
    if source.name != "source.jsonl" or not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise DataValidationError("public mixture manifest must be an object")
        manifest = PublicMixtureManifest.from_dict(value)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(
            f"invalid public mixture manifest {manifest_path}: {exc}"
        ) from exc

    _verify_mixture_manifest_recipe(manifest)
    actual_sha256 = sha256_file(source)
    if actual_sha256 != manifest.source_sha256:
        raise DataValidationError(
            f"public mixture source hash mismatch: expected {manifest.source_sha256}, "
            f"got {actual_sha256}"
        )
    return manifest


def _load_components(
    source_paths: list[str | Path],
) -> dict[str, tuple[Path, PublicSourceManifest]]:
    components: dict[str, tuple[Path, PublicSourceManifest]] = {}
    for value in source_paths:
        path = Path(value)
        manifest = load_public_source_manifest(path)
        if manifest is None:
            raise DataValidationError(
                f"public mixture input is not a materialized public source: {path}"
            )
        if manifest.recipe_id in components:
            raise DataValidationError(
                f"duplicate public mixture component: {manifest.recipe_id}"
            )
        components[manifest.recipe_id] = (path, manifest)
    return components


def _write_round_robin(
    components: list[tuple[Path, PublicSourceManifest]],
    path: Path,
    *,
    annotate_component: bool = False,
) -> int:
    expected_counts = [manifest.records for _, manifest in components]
    actual_counts = [0] * len(components)
    seen_ids: set[str] = set()
    total = 0
    try:
        with ExitStack() as stack:
            handles = [
                stack.enter_context(source.open("r", encoding="utf-8"))
                for source, _ in components
            ]
            iterators = [iter(handle) for handle in handles]
            active = [True] * len(iterators)
            with path.open("x", encoding="utf-8") as output:
                while any(active):
                    for index, iterator in enumerate(iterators):
                        if not active[index]:
                            continue
                        try:
                            line = next(iterator)
                        except StopIteration:
                            active[index] = False
                            continue
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        record_id = record.get("id")
                        if not isinstance(record_id, str) or not record_id:
                            raise DataValidationError(
                                "public mixture component contains an invalid record ID"
                            )
                        if record_id in seen_ids:
                            raise DataValidationError(
                                f"public mixture contains duplicate record ID: "
                                f"{record_id}"
                            )
                        seen_ids.add(record_id)
                        record["language"] = components[index][1].language
                        if annotate_component:
                            record["source_recipe_id"] = (
                                components[index][1].recipe_id
                            )
                        output.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        actual_counts[index] += 1
                        total += 1
                output.flush()
                os.fsync(output.fileno())
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DataValidationError(f"cannot write public mixture {path}: {exc}") from exc

    if actual_counts != expected_counts:
        raise DataValidationError(
            f"public mixture component counts changed: expected {expected_counts}, "
            f"got {actual_counts}"
        )
    return total


def _verify_mixture_manifest_recipe(manifest: PublicMixtureManifest) -> None:
    recipe = load_public_mixture_recipe(manifest.mixture_id)
    component_manifests = tuple(
        PublicSourceManifest.from_dict(component) for component in manifest.components
    )
    for component in component_manifests:
        verify_public_source_manifest_recipe(component)

    actual_ids = tuple(component.recipe_id for component in component_manifests)
    expected_records = sum(component.records for component in component_manifests)
    mismatches = []
    if actual_ids != recipe.components:
        mismatches.append("components")
    if manifest.strategy != recipe.strategy:
        mismatches.append("strategy")
    if manifest.annotate_component != recipe.annotate_component:
        mismatches.append("annotate_component")
    if manifest.records != expected_records:
        mismatches.append("records")
    if manifest.license != _license_expression(component_manifests):
        mismatches.append("license")
    if manifest.languages != _languages(component_manifests):
        mismatches.append("languages")
    if (
        recipe.expected_source_sha256 is not None
        and manifest.source_sha256 != recipe.expected_source_sha256
    ):
        mismatches.append("source_sha256")
    if mismatches:
        raise DataValidationError(
            "public mixture manifest does not match installed recipe "
            f"{manifest.mixture_id}: {', '.join(mismatches)}"
        )


def _license_expression(
    manifests: list[PublicSourceManifest] | tuple[PublicSourceManifest, ...],
) -> str:
    return " AND ".join(sorted({manifest.license for manifest in manifests}))


def _languages(
    manifests: list[PublicSourceManifest] | tuple[PublicSourceManifest, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(manifest.language for manifest in manifests))


def _load_mixture_recipe_resource(resource: Any) -> PublicMixtureRecipe:
    try:
        value = yaml.safe_load(resource.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"cannot load public mixture recipe {resource}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ConfigError(f"public mixture recipe root must be a mapping: {resource}")
    return PublicMixtureRecipe.from_dict(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(
                value,
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
            f"cannot write public mixture manifest: {exc}"
        ) from exc
