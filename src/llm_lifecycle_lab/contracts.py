"""Versioned contracts shared by lifecycle stages."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from llm_lifecycle_lab.exceptions import ContractError

SCHEMA_VERSION = "1.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[A-Za-z0-9.+-]*)$")


class ModelRoute(StrEnum):
    """Supported model families with isolated checkpoint histories."""

    NATIVE = "native"
    QWEN3_TRANSFER = "qwen3-transfer"


class RunProfile(StrEnum):
    """Execution scales and acceptance levels within a model family."""

    SMOKE = "smoke"
    LEARN = "learn"
    REPRODUCE = "reproduce"


class Stage(StrEnum):
    """Lifecycle stages understood by the configuration contract."""

    DATA = "data"
    TOKENIZER = "tokenizer"
    PRETRAIN = "pretrain"
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    EVALUATION = "evaluation"
    EXPORT = "export"


class RecordKind(StrEnum):
    """Supported JSONL record shapes."""

    PRETRAIN = "pretrain"
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"


def utc_now() -> str:
    """Return a stable UTC timestamp for persisted metadata."""

    return datetime.now(UTC).isoformat(timespec="seconds")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")


def _validate_sha256(value: str, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ContractError(f"{field_name} must be a lowercase SHA-256 digest")


def _to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            item.name: _to_primitive(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_primitive(item) for item in value]
    return value


class JsonContract:
    """Mixin for deterministic JSON serialization."""

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class FileFingerprint(JsonContract):
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _require_text(self.path, "path")
        _validate_sha256(self.sha256, "sha256")
        if self.size_bytes < 0:
            raise ContractError("size_bytes must be non-negative")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FileFingerprint:
        return cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            size_bytes=int(data["size_bytes"]),
        )


@dataclass(frozen=True, slots=True)
class DataSplitManifest(JsonContract):
    name: str
    path: str
    sha256: str
    records: int
    groups: int

    def __post_init__(self) -> None:
        _require_text(self.name, "split.name")
        _require_text(self.path, "split.path")
        _validate_sha256(self.sha256, "split.sha256")
        if self.records < 0 or self.groups < 0:
            raise ContractError("split records and groups must be non-negative")
        split_path = Path(self.path)
        if split_path.is_absolute() or ".." in split_path.parts:
            raise ContractError("split.path must remain inside the dataset directory")
        if self.groups > self.records:
            raise ContractError("split groups cannot exceed record count")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataSplitManifest:
        return cls(
            name=str(data["name"]),
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            records=int(data["records"]),
            groups=int(data["groups"]),
        )


@dataclass(frozen=True, slots=True)
class DataManifest(JsonContract):
    dataset_id: str
    record_kind: RecordKind
    source: FileFingerprint
    splits: tuple[DataSplitManifest, ...]
    split_seed: int
    group_by: str
    license: str
    processing_steps: tuple[str, ...] = ()
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.dataset_id, "dataset_id")
        _require_text(self.group_by, "group_by")
        _require_text(self.license, "license")
        if not self.splits:
            raise ContractError("data manifest must contain at least one split")
        names = [split.name for split in self.splits]
        if len(names) != len(set(names)):
            raise ContractError("data manifest contains duplicate split names")
        if set(names) != {"train", "dev", "test"}:
            raise ContractError(
                "data manifest must contain train, dev, and test splits"
            )
        if self.split_seed < 0:
            raise ContractError("split_seed must be non-negative")
        object.__setattr__(
            self,
            "source_metadata",
            _freeze_mapping(self.source_metadata, "data.source_metadata"),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(
                f"unsupported data manifest schema_version: {self.schema_version}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataManifest:
        return cls(
            dataset_id=str(data["dataset_id"]),
            record_kind=RecordKind(data["record_kind"]),
            source=FileFingerprint.from_dict(data["source"]),
            splits=tuple(DataSplitManifest.from_dict(item) for item in data["splits"]),
            split_seed=int(data["split_seed"]),
            group_by=str(data["group_by"]),
            license=str(data["license"]),
            processing_steps=tuple(
                str(item) for item in data.get("processing_steps", ())
            ),
            source_metadata=_mapping(
                data.get("source_metadata", {}),
                "data.source_metadata",
            ),
            created_at=str(data["created_at"]),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class TokenizerManifest(JsonContract):
    tokenizer_id: str
    revision: str
    vocab_size: int
    content_sha256: str
    source_data_sha256: str
    special_tokens: Mapping[str, int]
    trainer_config: Mapping[str, Any]
    chat_template: str | None = None
    chat_template_version: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.tokenizer_id, "tokenizer_id")
        _require_text(self.revision, "tokenizer.revision")
        _validate_sha256(self.content_sha256, "tokenizer.content_sha256")
        _validate_sha256(
            self.source_data_sha256,
            "tokenizer.source_data_sha256",
        )
        if self.vocab_size <= 0:
            raise ContractError("tokenizer.vocab_size must be positive")
        object.__setattr__(
            self,
            "special_tokens",
            _freeze_mapping(self.special_tokens, "tokenizer.special_tokens"),
        )
        object.__setattr__(
            self,
            "trainer_config",
            _freeze_mapping(self.trainer_config, "tokenizer.trainer_config"),
        )
        token_ids = list(self.special_tokens.values())
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id >= self.vocab_size
            for token_id in token_ids
        ):
            raise ContractError(
                "tokenizer special token IDs must be valid vocabulary indices"
            )
        if len(token_ids) != len(set(token_ids)):
            raise ContractError("tokenizer special token IDs must be unique")
        if (self.chat_template is None) != (self.chat_template_version is None):
            raise ContractError(
                "tokenizer chat_template and chat_template_version "
                "must be declared together"
            )
        if self.chat_template is not None:
            _require_text(self.chat_template, "tokenizer.chat_template")
            _require_text(
                self.chat_template_version,
                "tokenizer.chat_template_version",
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(
                f"unsupported tokenizer schema_version: {self.schema_version}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TokenizerManifest:
        return cls(
            tokenizer_id=str(data["tokenizer_id"]),
            revision=str(data["revision"]),
            vocab_size=int(data["vocab_size"]),
            content_sha256=str(data["content_sha256"]),
            source_data_sha256=str(data["source_data_sha256"]),
            special_tokens={
                str(token): int(token_id)
                for token, token_id in data["special_tokens"].items()
            },
            trainer_config=_mapping(
                data["trainer_config"],
                "tokenizer.trainer_config",
            ),
            chat_template=_optional_text(data.get("chat_template")),
            chat_template_version=_optional_text(data.get("chat_template_version")),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class ModelMetadata(JsonContract):
    model_route: ModelRoute
    provider: str
    model_id: str
    architecture: str
    upstream_revision: str | None = None
    weights_sha256: str | None = None
    tokenizer_revision: str | None = None
    chat_template_version: str | None = None
    parameter_count: int | None = None
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.provider, "model.provider")
        _require_text(self.model_id, "model.model_id")
        _require_text(self.architecture, "model.architecture")
        if self.parameter_count is not None and self.parameter_count <= 0:
            raise ContractError("model.parameter_count must be positive")
        if self.weights_sha256 is not None:
            _validate_sha256(self.weights_sha256, "model.weights_sha256")
        if self.model_route is ModelRoute.QWEN3_TRANSFER:
            if self.provider != "huggingface":
                raise ContractError("qwen3-transfer requires provider=huggingface")
            _require_text(self.upstream_revision or "", "model.upstream_revision")
            _require_text(self.tokenizer_revision or "", "model.tokenizer_revision")
            if not _COMMIT_PATTERN.fullmatch(self.upstream_revision or ""):
                raise ContractError(
                    "Qwen upstream_revision must be a 40-character commit SHA"
                )
            if not _COMMIT_PATTERN.fullmatch(self.tokenizer_revision or ""):
                raise ContractError(
                    "Qwen tokenizer_revision must be a 40-character commit SHA"
                )
        elif self.provider != "native":
            raise ContractError("native model metadata requires provider=native")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelMetadata:
        return cls(
            model_route=ModelRoute(data["model_route"]),
            provider=str(data["provider"]),
            model_id=str(data["model_id"]),
            architecture=str(data["architecture"]),
            upstream_revision=_optional_text(data.get("upstream_revision")),
            weights_sha256=_optional_text(data.get("weights_sha256")),
            tokenizer_revision=_optional_text(data.get("tokenizer_revision")),
            chat_template_version=_optional_text(data.get("chat_template_version")),
            parameter_count=_optional_int(data.get("parameter_count")),
            capabilities=tuple(str(item) for item in data.get("capabilities", ())),
        )


@dataclass(frozen=True, slots=True)
class ModelManifest(JsonContract):
    metadata: ModelMetadata
    config_sha256: str
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_sha256(self.config_sha256, "model_manifest.config_sha256")


@dataclass(frozen=True, slots=True)
class RunConfig(JsonContract):
    model_route: ModelRoute
    run_profile: RunProfile
    stage: Stage
    model: Mapping[str, Any]
    seed: int = 42
    output_dir: str = "runs"
    data: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _freeze_mapping(self.model, "model"))
        object.__setattr__(self, "data", _freeze_mapping(self.data, "data"))
        object.__setattr__(
            self,
            "training",
            _freeze_mapping(self.training, "training"),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(
                f"unsupported config schema_version: {self.schema_version}"
            )
        if self.seed < 0:
            raise ContractError("seed must be non-negative")
        _require_text(self.output_dir, "output_dir")
        model_id = self.model.get("model_id")
        _require_text(model_id, "model.model_id")
        self._validate_route_constraints()

    def _validate_route_constraints(self) -> None:
        if (
            self.model_route is ModelRoute.QWEN3_TRANSFER
            and self.run_profile is RunProfile.SMOKE
        ):
            raise ContractError("run_profile=smoke requires model_route=native")
        if self.model_route is ModelRoute.QWEN3_TRANSFER and self.stage in {
            Stage.PRETRAIN,
            Stage.TOKENIZER,
        }:
            raise ContractError(
                "qwen3-transfer cannot run tokenizer training or pretraining"
            )
        provider = self.model.get("provider")
        if self.model_route is ModelRoute.QWEN3_TRANSFER:
            if provider != "huggingface":
                raise ContractError(
                    "qwen3-transfer requires model.provider=huggingface"
                )
            if self.model["model_id"] != "Qwen/Qwen3-0.6B-Base":
                raise ContractError(
                    "qwen3-transfer requires model.model_id=Qwen/Qwen3-0.6B-Base"
                )
            revision = self.model.get("revision")
            tokenizer_revision = self.model.get("tokenizer_revision")
            _require_text(revision, "model.revision")
            _require_text(tokenizer_revision, "model.tokenizer_revision")
            if revision != tokenizer_revision:
                raise ContractError("Qwen model and tokenizer revisions must match")
            if not _COMMIT_PATTERN.fullmatch(str(revision)):
                raise ContractError(
                    "Qwen model.revision must be a 40-character commit SHA"
                )
            transformers_version = self.model.get("transformers_version")
            _require_text(transformers_version, "model.transformers_version")
            if not _VERSION_PATTERN.fullmatch(str(transformers_version)):
                raise ContractError(
                    "model.transformers_version must be an exact version"
                )
        elif provider != "native":
            raise ContractError("native route requires model.provider=native")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        allowed = {
            "schema_version",
            "model_route",
            "run_profile",
            "stage",
            "model",
            "seed",
            "output_dir",
            "data",
            "training",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ContractError(f"unknown run config fields: {', '.join(unknown)}")
        try:
            return cls(
                schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
                model_route=ModelRoute(data["model_route"]),
                run_profile=RunProfile(data["run_profile"]),
                stage=Stage(data["stage"]),
                model=_mapping(data["model"], "model"),
                seed=int(data.get("seed", 42)),
                output_dir=str(data.get("output_dir", "runs")),
                data=_mapping(data.get("data", {}), "data"),
                training=_mapping(data.get("training", {}), "training"),
            )
        except KeyError as exc:
            raise ContractError(
                f"missing required run config field: {exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid run config value: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RunManifest(JsonContract):
    run_id: str
    config_sha256: str
    model_route: ModelRoute
    run_profile: RunProfile
    stage: Stage
    status: str = "created"
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        _validate_sha256(self.config_sha256, "run.config_sha256")
        if self.status not in {"created", "running", "completed", "failed"}:
            raise ContractError(f"invalid run status: {self.status}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        return cls(
            run_id=str(data["run_id"]),
            config_sha256=str(data["config_sha256"]),
            model_route=ModelRoute(data["model_route"]),
            run_profile=RunProfile(data["run_profile"]),
            stage=Stage(data["stage"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class CheckpointMetadata(JsonContract):
    checkpoint_id: str
    run_id: str
    model_route: ModelRoute
    stage: Stage
    step: int
    tokenizer_sha256: str
    config_sha256: str
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.checkpoint_id, "checkpoint_id")
        _require_text(self.run_id, "checkpoint.run_id")
        _validate_sha256(self.tokenizer_sha256, "checkpoint.tokenizer_sha256")
        _validate_sha256(self.config_sha256, "checkpoint.config_sha256")
        if self.step < 0:
            raise ContractError("checkpoint.step must be non-negative")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckpointMetadata:
        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            run_id=str(data["run_id"]),
            model_route=ModelRoute(data["model_route"]),
            stage=Stage(data["stage"]),
            step=int(data["step"]),
            tokenizer_sha256=str(data["tokenizer_sha256"]),
            config_sha256=str(data["config_sha256"]),
            created_at=str(data["created_at"]),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class EvaluationReport(JsonContract):
    run_id: str
    suite: str
    metrics: Mapping[str, float]
    sample_count: int
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.run_id, "evaluation.run_id")
        _require_text(self.suite, "evaluation.suite")
        if self.sample_count < 0:
            raise ContractError("evaluation.sample_count must be non-negative")
        object.__setattr__(
            self,
            "metrics",
            _freeze_mapping(self.metrics, "evaluation.metrics"),
        )


@dataclass(frozen=True, slots=True)
class ExportManifest(JsonContract):
    run_id: str
    format: str
    files: tuple[FileFingerprint, ...]
    verification_passed: bool
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.run_id, "export.run_id")
        _require_text(self.format, "export.format")
        if not self.files:
            raise ContractError("export manifest must list at least one file")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    return _freeze_mapping(value, field_name)


def _freeze_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field_name} must be a mapping")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ContractError(f"{field_name} keys must be strings")
        frozen[key] = _freeze_value(item, f"{field_name}.{key}")
    return MappingProxyType(frozen)


def _freeze_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value, field_name)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{field_name} must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ContractError(
        f"{field_name} contains unsupported value type {type(value).__name__}"
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
