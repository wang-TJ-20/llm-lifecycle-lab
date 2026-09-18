"""检查冻结的基线输入、训练条件和已完成 run 的验收结果。

Verify frozen baseline inputs, training prerequisites, and completed runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab.config import (
    canonical_json,
    config_sha256,
    load_mapping,
    load_run_config,
)
from llm_lifecycle_lab.contracts import (
    SCHEMA_VERSION,
    CheckpointMetadata,
    JsonContract,
    RunConfig,
    RunManifest,
    utc_now,
)
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.packing import (
    load_packed_pretraining_manifest,
    verify_packed_pretraining_manifest,
)
from llm_lifecycle_lab.data.prepare import load_data_manifest, verify_data_manifest
from llm_lifecycle_lab.doctor.result import CheckResult, CheckStatus
from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError, ContractError
from llm_lifecycle_lab.model.native import (
    NativeModelConfig,
    load_native_model_config,
)
from llm_lifecycle_lab.provenance import capture_runtime_provenance, source_tree_sha256
from llm_lifecycle_lab.tokenizer import NativeTokenizer
from llm_lifecycle_lab.training.engine import EngineConfig

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReferenceRunReport(JsonContract):
    reference_id: str
    run_path: str
    checks: tuple[CheckResult, ...]
    scope: str = "completed-run"
    created_at: str = field(default_factory=utc_now)

    @property
    def has_failures(self) -> bool:
        return any(check.status is CheckStatus.FAIL for check in self.checks)

    @property
    def exit_code(self) -> int:
        return 1 if self.has_failures else 0

    @property
    def counts(self) -> dict[str, int]:
        return {
            status.value: sum(check.status is status for check in self.checks)
            for status in CheckStatus
        }

    def to_dict(self) -> dict[str, Any]:
        value = JsonContract.to_dict(self)
        value["counts"] = self.counts
        value["ok"] = not self.has_failures
        return value


@dataclass(frozen=True, slots=True)
class _FrozenBaseline:
    execution_sha256: str
    source_sha256: str
    packed_manifest_sha256: str
    python_major_minor: str
    packages: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _ReferenceSpec:
    reference_id: str
    pipeline_config: Path
    expected_data_manifest_sha256: str
    expected_tokenizer_sha256: str
    expected_model_config_sha256: str
    expected_packed_sequence_length: int
    expected_packed_array_sha256: tuple[tuple[str, str], ...]
    expected_budget_mode: str
    expected_global_step: int
    expected_target_train_tokens: int
    minimum_target_token_coverage: float
    maximum_target_token_coverage: float
    required_training_metric_keys: tuple[str, ...]
    required_evaluation_metric_keys: tuple[str, ...]
    evaluation_suites: tuple[str, ...]
    platform_system: str
    device_type: str
    minimum_device_memory_gib: float
    require_clean_git: bool
    require_eval_improvement: bool
    freeze: _FrozenBaseline | None = None
    require_final_language_improvement: bool = False
    expected_evaluation_samples: int | None = None

    def __post_init__(self) -> None:
        if self.expected_budget_mode not in {
            "max_steps",
            "max_train_tokens",
            "num_epochs",
        }:
            raise ConfigError("reference expected_budget_mode is unsupported")
        if self.expected_global_step <= 0 or self.expected_target_train_tokens <= 0:
            raise ConfigError("reference step and token targets must be positive")
        if self.expected_packed_sequence_length < 2:
            raise ConfigError("reference packed sequence length must be at least 2")
        for name, digest in (
            ("data manifest", self.expected_data_manifest_sha256),
            ("tokenizer", self.expected_tokenizer_sha256),
            ("model config", self.expected_model_config_sha256),
            *self.expected_packed_array_sha256,
        ):
            if not _SHA256_PATTERN.fullmatch(digest):
                raise ConfigError(f"reference {name} SHA-256 is invalid")
        if not self.expected_packed_array_sha256:
            raise ConfigError("reference requires packed array SHA-256 values")
        if (
            not math.isfinite(self.minimum_target_token_coverage)
            or not math.isfinite(self.maximum_target_token_coverage)
            or self.minimum_target_token_coverage <= 0
            or self.maximum_target_token_coverage < self.minimum_target_token_coverage
        ):
            raise ConfigError("reference token coverage bounds are invalid")
        if not self.required_training_metric_keys:
            raise ConfigError("reference requires training metric keys")
        if not self.required_evaluation_metric_keys:
            raise ConfigError("reference requires evaluation metric keys")
        if not self.evaluation_suites:
            raise ConfigError("reference requires evaluation suites")
        if not self.platform_system.strip() or not self.device_type.strip():
            raise ConfigError("reference platform and device type must not be empty")
        if (
            not math.isfinite(self.minimum_device_memory_gib)
            or self.minimum_device_memory_gib < 0
        ):
            raise ConfigError("reference minimum device memory is invalid")
        if self.expected_evaluation_samples is not None and (
            type(self.expected_evaluation_samples) is not int
            or self.expected_evaluation_samples <= 0
        ):
            raise ConfigError("reference expected_evaluation_samples must be positive")


def reference_execution_sha256(config: RunConfig, *, workdir: str | Path = ".") -> str:
    """Pin effective model/trainer defaults as well as the pipeline YAML values."""
    model_path = _input_path(config.model, "config", Path(workdir).resolve())
    execution = {
        "pipeline": config.to_dict(),
        "model": asdict(load_native_model_config(model_path)),
        "training": asdict(EngineConfig.from_dict(config.training)),
    }
    return hashlib.sha256(canonical_json(execution).encode("utf-8")).hexdigest()


def verify_reference_inputs(
    spec_path: str | Path,
    *,
    workdir: str | Path = ".",
    config: RunConfig | None = None,
    check_runtime: bool = True,
) -> ReferenceRunReport:
    """Read-only preflight; an inputs-only success is not permission to train."""
    root = Path(workdir).resolve()
    scope = "preflight" if check_runtime else "inputs-only"
    spec = _load_reference_spec(Path(spec_path), root=root)
    expected = load_run_config(spec.pipeline_config)
    checks = _configuration_checks(spec, expected, root)
    if config is not None:
        checks.append(
            _check(
                "selected-pipeline",
                config_sha256(config) == config_sha256(expected),
                "selected pipeline matches the frozen baseline",
                "selected pipeline differs from the baseline; no run was started",
            )
        )
    if any(check.status is CheckStatus.FAIL for check in checks):
        return ReferenceRunReport(spec.reference_id, str(root), tuple(checks), scope)

    data_path = _input_path(expected.data, "manifest", root)
    packed_path = _input_path(expected.data, "packed_manifest", root)
    tokenizer_path = _input_path(expected.model, "tokenizer", root)
    model_path = _input_path(expected.model, "config", root)
    data = load_data_manifest(data_path)
    packed = load_packed_pretraining_manifest(packed_path)
    tokenizer = NativeTokenizer.from_directory(tokenizer_path)
    model = load_native_model_config(model_path)
    failures = verify_data_manifest(data_path)
    failures.extend(verify_packed_pretraining_manifest(packed_path))
    provenance_ok = (
        not failures
        and sha256_file(data_path) == spec.expected_data_manifest_sha256
        and tokenizer.manifest.content_sha256 == spec.expected_tokenizer_sha256
        and packed.data_manifest_sha256 == spec.expected_data_manifest_sha256
        and packed.tokenizer_sha256 == spec.expected_tokenizer_sha256
        and sha256_file(model_path) == spec.expected_model_config_sha256
        and tokenizer.vocab_size == model.vocab_size
        and model.model_id == expected.model.get("model_id")
        and data.dataset_id == packed.dataset_id
        and packed.sequence_length == spec.expected_packed_sequence_length
        and _packed_array_sha256(packed.to_dict())
        == dict(spec.expected_packed_array_sha256)
    )
    if spec.freeze is not None:
        provenance_ok = provenance_ok and (
            sha256_file(packed_path) == spec.freeze.packed_manifest_sha256
        )
    checks.append(
        _check(
            "input-artifacts",
            provenance_ok,
            "prepared files, tokenizer and all packed arrays match the baseline",
            "baseline input mismatch: "
            + "; ".join(failures or ["hash or metadata differs"]),
        )
    )
    engine = EngineConfig.from_dict(expected.training)
    train_split = next(
        (split for split in packed.splits if split.name == "train"), None
    )
    if train_split is None:
        raise ConfigError("reference packed data requires a train split")
    _, budget = engine.resolve_budget(
        examples_per_epoch=train_split.examples,
        supervised_tokens_per_epoch=train_split.supervised_tokens,
    )
    checks.append(
        _check(
            "input-budget",
            budget.mode == spec.expected_budget_mode
            and budget.max_steps == spec.expected_global_step
            and budget.target_train_tokens == spec.expected_target_train_tokens
            and engine.sequence_length == packed.sequence_length
            and engine.sequence_length <= model.max_sequence_length,
            "resolved budget matches the frozen baseline",
            "resolved budget or sequence length differs from the baseline",
            budget.to_dict(),
        )
    )
    if spec.freeze is not None:
        checks.append(
            _check(
                "source-lock",
                source_tree_sha256(root) == spec.freeze.source_sha256,
                "Python source matches the frozen baseline",
                "Python source differs from the baseline; use its original checkout",
            )
        )
    if check_runtime:
        runtime = capture_runtime_provenance(
            workdir=root, device=torch.device(engine.device)
        )
        checks.append(_runtime_check(spec, runtime))
        if engine.device == "cuda":
            bf16_ok = torch.cuda.is_available() and (
                engine.dtype != "bfloat16" or torch.cuda.is_bf16_supported()
            )
            checks.append(
                _check(
                    "cuda-compute",
                    bf16_ok,
                    "CUDA supports the requested training dtype",
                    "CUDA or the required BF16 support is unavailable",
                )
            )
        if spec.freeze is not None:
            checks.append(_frozen_runtime_check(spec.freeze, runtime))
    return ReferenceRunReport(
        spec.reference_id,
        str(root),
        tuple(checks),
        scope,
    )


def verify_reference_run(
    spec_path: str | Path,
    run_path: str | Path,
    *,
    workdir: str | Path = ".",
) -> ReferenceRunReport:
    """Verify one completed run against a versioned acceptance specification."""

    root = Path(workdir).resolve()
    spec = _load_reference_spec(Path(spec_path), root=root)
    run = Path(run_path)
    if not run.is_absolute():
        run = root / run
    checks: list[CheckResult] = []

    required_files = (
        "run_manifest.json",
        "resolved_config.yaml",
        "data_snapshot.json",
        "data_provenance.json",
        "packed_data_snapshot.json",
        "tokenizer_manifest.json",
        "tokenizer/tokenizer.json",
        "model_config.json",
        "model_manifest.json",
        "training_budget.json",
        "training_result.json",
        "runtime_environment.json",
        "metrics.jsonl",
        "latest_checkpoint.json",
    )
    missing = [name for name in required_files if not (run / name).is_file()]
    checks.append(
        _check(
            "required-artifacts",
            not missing,
            "all required run artifacts are present",
            "missing run artifacts: " + ", ".join(missing),
            {"missing": missing},
        )
    )
    if missing:
        return ReferenceRunReport(
            reference_id=spec.reference_id,
            run_path=str(run),
            checks=tuple(checks),
        )

    expected_config = load_run_config(spec.pipeline_config)
    checks.extend(_configuration_checks(spec, expected_config, root))
    if any(check.status is CheckStatus.FAIL for check in checks):
        return ReferenceRunReport(spec.reference_id, str(run), tuple(checks))
    run_manifest = _load_contract(
        run / "run_manifest.json",
        RunManifest.from_dict,
        "run manifest",
    )
    resolved_config = load_run_config(run / "resolved_config.yaml")
    expected_config_sha256 = config_sha256(expected_config)
    checks.append(
        _check(
            "run-contract",
            (
                run_manifest.status == "completed"
                and run_manifest.model_route is expected_config.model_route
                and run_manifest.run_profile is expected_config.run_profile
                and run_manifest.stage is expected_config.stage
                and run_manifest.config_sha256 == expected_config_sha256
                and config_sha256(resolved_config) == expected_config_sha256
            ),
            "run status, route, profile, stage, and config hash match",
            "run contract does not match the reference pipeline",
            {
                "status": run_manifest.status,
                "config_sha256": run_manifest.config_sha256,
                "expected_config_sha256": expected_config_sha256,
            },
        )
    )

    data_snapshot = _load_json(run / "data_snapshot.json", "data snapshot")
    data_provenance = _load_json(
        run / "data_provenance.json",
        "data provenance",
    )
    packed_snapshot = _load_json(
        run / "packed_data_snapshot.json",
        "packed data snapshot",
    )
    tokenizer_manifest = _load_json(
        run / "tokenizer_manifest.json",
        "tokenizer manifest",
    )
    model_snapshot = _load_json(run / "model_config.json", "model config")
    model_manifest = _load_json(run / "model_manifest.json", "model manifest")
    data_sha256 = sha256_file(run / "data_snapshot.json")
    tokenizer_sha256 = sha256_file(run / "tokenizer/tokenizer.json")
    model_config_path = Path(str(expected_config.model.get("config", "")))
    if not model_config_path.is_absolute():
        model_config_path = root / model_config_path
    expected_model_config = load_native_model_config(model_config_path)
    packed_array_sha256 = _packed_array_sha256(packed_snapshot)
    provenance_ok = (
        packed_snapshot.get("data_manifest_sha256") == data_sha256
        and packed_snapshot.get("tokenizer_sha256") == tokenizer_sha256
        and tokenizer_manifest.get("content_sha256") == tokenizer_sha256
        and data_provenance.get("training_data_manifest_sha256") == data_sha256
        and data_provenance.get("tokenizer_source_data_manifest_sha256")
        == tokenizer_manifest.get("source_data_sha256")
        and data_provenance.get("tokenizer_sha256") == tokenizer_sha256
        and data_snapshot.get("dataset_id") == packed_snapshot.get("dataset_id")
        and data_sha256 == spec.expected_data_manifest_sha256
        and tokenizer_sha256 == spec.expected_tokenizer_sha256
        and sha256_file(model_config_path) == spec.expected_model_config_sha256
        and model_manifest.get("config_sha256") == spec.expected_model_config_sha256
        and NativeModelConfig.from_dict(model_snapshot) == expected_model_config
        and _as_int(packed_snapshot.get("sequence_length"))
        == spec.expected_packed_sequence_length
        and packed_array_sha256 == dict(spec.expected_packed_array_sha256)
    )
    if spec.freeze is not None:
        provenance_ok = provenance_ok and (
            sha256_file(run / "packed_data_snapshot.json")
            == spec.freeze.packed_manifest_sha256
        )
    checks.append(
        _check(
            "data-provenance",
            provenance_ok,
            "data, tokenizer, and packed snapshots form a consistent hash chain",
            "data, tokenizer, and packed snapshot hashes are inconsistent",
            {
                "data_manifest_sha256": data_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "model_config_sha256": model_manifest.get("config_sha256"),
                "packed_arrays": packed_array_sha256,
            },
        )
    )

    budget = _load_json(run / "training_budget.json", "training budget")
    result = _load_json(run / "training_result.json", "training result")
    budget_ok = (
        budget.get("mode") == spec.expected_budget_mode
        and _as_int(budget.get("max_steps")) == spec.expected_global_step
        and _as_int(budget.get("target_train_tokens"))
        == spec.expected_target_train_tokens
        and _as_int(result.get("global_step")) == spec.expected_global_step
        and _as_int(result.get("target_train_tokens"))
        == spec.expected_target_train_tokens
    )
    checks.append(
        _check(
            "training-budget",
            budget_ok,
            "resolved epoch budget and completed step count match",
            "training budget or completed step count does not match",
            {
                "mode": budget.get("mode"),
                "global_step": result.get("global_step"),
                "target_train_tokens": result.get("target_train_tokens"),
            },
        )
    )
    coverage = _as_float(result.get("target_token_coverage"))
    result_ok = (
        coverage is not None
        and spec.minimum_target_token_coverage
        <= coverage
        <= spec.maximum_target_token_coverage
        and _positive_finite(result.get("elapsed_seconds"))
        and _positive_finite(result.get("final_loss"))
        and _positive_finite(result.get("best_eval_loss"))
    )
    checks.append(
        _check(
            "training-result",
            result_ok,
            "training result is finite and meets token coverage bounds",
            "training result is non-finite or outside token coverage bounds",
            {
                "target_token_coverage": coverage,
                "minimum": spec.minimum_target_token_coverage,
                "maximum": spec.maximum_target_token_coverage,
            },
        )
    )

    metrics = _load_metrics(run / "metrics.jsonl")
    baseline = metrics[0] if metrics else {}
    final_metrics = next(
        (
            item
            for item in reversed(metrics)
            if _as_int(item.get("step")) == spec.expected_global_step
        ),
        {},
    )
    missing_metrics = [
        key
        for key in spec.required_training_metric_keys
        if not _finite(final_metrics.get(key))
    ]
    metrics_ok = (
        baseline.get("event") == "baseline"
        and not missing_metrics
        and _language_loss_is_consistent(final_metrics)
    )
    if spec.require_eval_improvement:
        baseline_loss = _as_float(baseline.get("eval_loss"))
        best_loss = _as_float(result.get("best_eval_loss"))
        metrics_ok = (
            metrics_ok
            and baseline_loss is not None
            and best_loss is not None
            and best_loss < baseline_loss
        )
    if spec.require_final_language_improvement:
        metrics_ok = metrics_ok and _final_language_improved(baseline, final_metrics)
    checks.append(
        _check(
            "metrics",
            metrics_ok,
            "baseline, final metrics, and language buckets are valid",
            "metrics are missing, non-finite, inconsistent, or did not improve",
            {"missing_metric_keys": missing_metrics},
        )
    )

    latest = _load_json(run / "latest_checkpoint.json", "latest checkpoint")
    checkpoint_path = _safe_child(run, latest.get("path"))
    checkpoint_ok = False
    if checkpoint_path is not None:
        metadata_path = checkpoint_path / "checkpoint_metadata.json"
        required_checkpoint_files = (
            metadata_path,
            checkpoint_path / "trainer_state.json",
            checkpoint_path / "optimizer_state.pt",
            checkpoint_path / "model/model.pt",
            checkpoint_path / "model/config.json",
        )
        if all(path.is_file() for path in required_checkpoint_files):
            metadata = _load_contract(
                metadata_path,
                CheckpointMetadata.from_dict,
                "checkpoint metadata",
            )
            checkpoint_ok = (
                metadata.run_id == run_manifest.run_id
                and metadata.step == spec.expected_global_step
                and metadata.config_sha256 == expected_config_sha256
                and metadata.tokenizer_sha256 == tokenizer_sha256
                and metadata.model_route is run_manifest.model_route
                and metadata.stage is run_manifest.stage
                and _as_int(latest.get("step")) == spec.expected_global_step
            )
    checks.append(
        _check(
            "checkpoint",
            checkpoint_ok,
            "final checkpoint is complete and compatible",
            "final checkpoint is missing, unsafe, or incompatible",
        )
    )

    missing_suites = _missing_evaluation_suites(
        run,
        run_id=run_manifest.run_id,
        suites=spec.evaluation_suites,
        required_metric_keys=spec.required_evaluation_metric_keys,
        expected_step=spec.expected_global_step,
        expected_samples=spec.expected_evaluation_samples,
    )
    checks.append(
        _check(
            "evaluations",
            not missing_suites,
            "all required dev/test evaluation reports are valid",
            "missing or invalid evaluation suites: " + ", ".join(missing_suites),
            {"missing_or_invalid": missing_suites},
        )
    )

    runtime = _load_json(
        run / "runtime_environment.json",
        "runtime environment",
    )
    checks.append(_runtime_check(spec, runtime))
    if spec.freeze is not None:
        checks.append(_frozen_runtime_check(spec.freeze, runtime))

    return ReferenceRunReport(
        reference_id=spec.reference_id,
        run_path=str(run),
        checks=tuple(checks),
    )


def _runtime_check(spec: _ReferenceSpec, runtime: Mapping[str, Any]) -> CheckResult:
    accelerator = _mapping(runtime.get("accelerator"))
    devices = accelerator.get("devices")
    device_memories = (
        [_as_int(_mapping(device).get("total_memory_bytes")) or 0 for device in devices]
        if isinstance(devices, list)
        else []
    )
    minimum_memory_bytes = int(spec.minimum_device_memory_gib * 1024**3)
    environment_ok = (
        _mapping(runtime.get("platform")).get("system") == spec.platform_system
        and accelerator.get("type") == spec.device_type
        and max(device_memories, default=0) >= minimum_memory_bytes
    )
    code = _mapping(runtime.get("code"))
    if spec.require_clean_git:
        environment_ok = (
            environment_ok
            and isinstance(code.get("commit"), str)
            and bool(_COMMIT_PATTERN.fullmatch(str(code["commit"])))
            and code.get("dirty") is False
        )
    if spec.freeze is not None and spec.device_type == "cuda":
        environment_ok = environment_ok and (
            accelerator.get("selected_device") in {"cuda", "cuda:0"}
            and bool(device_memories)
            and device_memories[0] >= minimum_memory_bytes
        )
    return _check(
        "runtime-provenance",
        environment_ok,
        "runtime platform, accelerator, and source state meet requirements",
        "runtime platform, accelerator, or source state is not publishable",
        {
            "platform": _mapping(runtime.get("platform")).get("system"),
            "device_type": accelerator.get("type"),
            "selected_device": accelerator.get("selected_device"),
            "first_device_memory_bytes": device_memories[0] if device_memories else 0,
            "maximum_device_memory_bytes": max(device_memories, default=0),
            "minimum_device_memory_bytes": minimum_memory_bytes,
            "git_commit": code.get("commit"),
            "git_dirty": code.get("dirty"),
        },
    )


def _load_reference_spec(path: Path, *, root: Path) -> _ReferenceSpec:
    value = load_mapping(path)
    if str(value.get("schema_version")) != SCHEMA_VERSION:
        raise ConfigError("reference spec has an unsupported schema_version")
    reference_id = value.get("reference_id")
    pipeline_value = value.get("pipeline_config")
    requirements = value.get("requirements")
    if not isinstance(reference_id, str) or not reference_id.strip():
        raise ConfigError("reference spec requires reference_id")
    if not isinstance(pipeline_value, str) or not pipeline_value.strip():
        raise ConfigError("reference spec requires pipeline_config")
    if not isinstance(requirements, Mapping):
        raise ConfigError("reference spec requires a requirements mapping")
    pipeline_path = Path(pipeline_value)
    if not pipeline_path.is_absolute():
        pipeline_path = root / pipeline_path
    try:
        return _ReferenceSpec(
            reference_id=reference_id,
            pipeline_config=pipeline_path,
            expected_data_manifest_sha256=_required_sha256(
                requirements["expected_data_manifest_sha256"],
                "expected_data_manifest_sha256",
            ),
            expected_tokenizer_sha256=_required_sha256(
                requirements["expected_tokenizer_sha256"],
                "expected_tokenizer_sha256",
            ),
            expected_model_config_sha256=_required_sha256(
                requirements["expected_model_config_sha256"],
                "expected_model_config_sha256",
            ),
            expected_packed_sequence_length=int(
                requirements["expected_packed_sequence_length"]
            ),
            expected_packed_array_sha256=_required_sha256_mapping(
                requirements["expected_packed_array_sha256"],
                "expected_packed_array_sha256",
            ),
            expected_budget_mode=str(requirements["expected_budget_mode"]),
            expected_global_step=int(requirements["expected_global_step"]),
            expected_target_train_tokens=int(
                requirements["expected_target_train_tokens"]
            ),
            minimum_target_token_coverage=float(
                requirements["minimum_target_token_coverage"]
            ),
            maximum_target_token_coverage=float(
                requirements["maximum_target_token_coverage"]
            ),
            required_training_metric_keys=_required_string_tuple(
                requirements["required_training_metric_keys"],
                "required_training_metric_keys",
            ),
            required_evaluation_metric_keys=_required_string_tuple(
                requirements["required_evaluation_metric_keys"],
                "required_evaluation_metric_keys",
            ),
            evaluation_suites=_required_string_tuple(
                requirements["evaluation_suites"],
                "evaluation_suites",
            ),
            platform_system=str(requirements["platform_system"]),
            device_type=str(requirements["device_type"]),
            minimum_device_memory_gib=float(requirements["minimum_device_memory_gib"]),
            require_clean_git=_required_bool(
                requirements["require_clean_git"],
                "require_clean_git",
            ),
            require_eval_improvement=_required_bool(
                requirements["require_eval_improvement"],
                "require_eval_improvement",
            ),
            freeze=_load_freeze(value.get("freeze")),
            require_final_language_improvement=_required_bool(
                requirements.get("require_final_language_improvement", False),
                "require_final_language_improvement",
            ),
            expected_evaluation_samples=requirements.get("expected_evaluation_samples"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid reference requirements: {exc}") from exc


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must contain a JSON object: {path}")
    return value


def _load_contract(
    path: Path,
    factory: Any,
    name: str,
) -> Any:
    try:
        return factory(_load_json(path, name))
    except (KeyError, TypeError, ValueError, ContractError) as exc:
        raise ArtifactError(f"invalid {name} {path}: {exc}") from exc


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read metrics {path}: {exc}") from exc
    if any(not isinstance(item, dict) for item in values):
        raise ArtifactError(f"metrics must contain JSON objects: {path}")
    return values


def _packed_array_sha256(
    packed_manifest: Mapping[str, Any],
) -> dict[str, str]:
    splits = packed_manifest.get("splits")
    if not isinstance(splits, list):
        return {}
    values: dict[str, str] = {}
    for split in splits:
        split_mapping = _mapping(split)
        for field_name in ("token_ids", "language_ids", "byte_weights"):
            array = _mapping(split_mapping.get(field_name))
            path = array.get("path")
            digest = array.get("sha256")
            if (
                not isinstance(path, str)
                or path in values
                or not isinstance(digest, str)
            ):
                return {}
            values[path] = digest
    return values


def _missing_evaluation_suites(
    run: Path,
    *,
    run_id: str,
    suites: tuple[str, ...],
    required_metric_keys: tuple[str, ...],
    expected_step: int,
    expected_samples: int | None = None,
) -> list[str]:
    missing: list[str] = []
    for suite in suites:
        report_path = run / "evaluations" / f"{suite}-step-{expected_step:08d}.json"
        if not report_path.is_file():
            missing.append(suite)
            continue
        report = _load_json(report_path, f"evaluation suite {suite}")
        metrics = _mapping(report.get("metrics"))
        if (
            report.get("run_id") != run_id
            or report.get("suite") != suite
            or (
                expected_samples is not None
                and report.get("sample_count") != expected_samples
            )
            or any(not _finite(metrics.get(key)) for key in required_metric_keys)
            or not _language_loss_is_consistent(metrics)
        ):
            missing.append(suite)
    return missing


def _input_path(values: Mapping[str, Any], name: str, root: Path) -> Path:
    value = values.get(name)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"reference pipeline requires {name}")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _configuration_checks(
    spec: _ReferenceSpec, config: RunConfig, root: Path
) -> list[CheckResult]:
    if spec.freeze is None:
        return []
    actual = reference_execution_sha256(config, workdir=root)
    return [
        _check(
            "execution-lock",
            actual == spec.freeze.execution_sha256,
            "pipeline and effective model/training defaults match the baseline",
            "pipeline or effective defaults changed; create a new experiment version",
            {"expected_sha256": spec.freeze.execution_sha256, "actual_sha256": actual},
        )
    ]


def _load_freeze(value: Any) -> _FrozenBaseline | None:
    if value is None:
        return None
    keys = {
        "execution_sha256",
        "source_sha256",
        "packed_manifest_sha256",
        "python_major_minor",
        "packages",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ConfigError(
            "reference freeze must declare exactly: " + ", ".join(sorted(keys))
        )
    version = value["python_major_minor"]
    packages = value["packages"]
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+", version):
        raise ConfigError("reference python_major_minor must be a major.minor string")
    if (
        not isinstance(packages, Mapping)
        or not packages
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in packages.items()
        )
    ):
        raise ConfigError("reference freeze packages must map names to version strings")
    return _FrozenBaseline(
        execution_sha256=_required_sha256(
            value["execution_sha256"], "execution_sha256"
        ),
        source_sha256=_required_sha256(value["source_sha256"], "source_sha256"),
        packed_manifest_sha256=_required_sha256(
            value["packed_manifest_sha256"], "packed_manifest_sha256"
        ),
        python_major_minor=version,
        packages=tuple(sorted(packages.items())),
    )


def _frozen_runtime_check(
    freeze: _FrozenBaseline, runtime: Mapping[str, Any]
) -> CheckResult:
    version = _mapping(runtime.get("python")).get("version", "")
    packages = _mapping(runtime.get("packages"))
    mismatches = [
        name
        for name, expected in freeze.packages
        if (
            str(packages.get(name, "")).split("+")[0]
            if name == "torch"
            else packages.get(name)
        )
        != expected
    ]
    matches = (
        isinstance(version, str)
        and ".".join(version.split(".")[:2]) == freeze.python_major_minor
        and not mismatches
        and _mapping(runtime.get("code")).get("source_sha256") == freeze.source_sha256
    )
    return _check(
        "frozen-runtime",
        matches,
        "Python, training packages and source snapshot match the baseline",
        "Python, training packages or source snapshot differ from the baseline",
        {
            "python": version,
            "expected_python": freeze.python_major_minor,
            "package_mismatches": {
                name: {"expected": expected, "actual": packages.get(name)}
                for name, expected in freeze.packages
                if name in mismatches
            },
            "source_sha256": _mapping(runtime.get("code")).get("source_sha256"),
            "expected_source_sha256": freeze.source_sha256,
        },
    )


def _final_language_improved(
    baseline: Mapping[str, Any], final: Mapping[str, Any]
) -> bool:
    for key in ("eval_loss", "eval_en_loss", "eval_zh_loss"):
        before, after = _as_float(baseline.get(key)), _as_float(final.get(key))
        if (
            before is None
            or after is None
            or not math.isfinite(before)
            or not math.isfinite(after)
            or not 0 < after < before
        ):
            return False
    return True


def _language_loss_is_consistent(metrics: Mapping[str, Any]) -> bool:
    loss = _as_float(metrics.get("eval_loss"))
    total = _as_float(metrics.get("eval_tokens"))
    en_loss = _as_float(metrics.get("eval_en_loss"))
    en_tokens = _as_float(metrics.get("eval_en_tokens"))
    zh_loss = _as_float(metrics.get("eval_zh_loss"))
    zh_tokens = _as_float(metrics.get("eval_zh_tokens"))
    values = (loss, total, en_loss, en_tokens, zh_loss, zh_tokens)
    if any(value is None for value in values):
        return False
    assert all(value is not None for value in values)
    if total <= 0 or en_tokens <= 0 or zh_tokens <= 0:
        return False
    if not math.isclose(total, en_tokens + zh_tokens, rel_tol=0, abs_tol=1e-6):
        return False
    weighted = (en_loss * en_tokens + zh_loss * zh_tokens) / total
    return math.isclose(loss, weighted, rel_tol=1e-6, abs_tol=1e-6)


def _safe_child(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    if path.is_symlink():
        return None
    return path


def _check(
    name: str,
    condition: bool,
    passed: str,
    failed: str,
    details: Mapping[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(
        name=name,
        status=CheckStatus.PASS if condition else CheckStatus.FAIL,
        message=passed if condition else failed,
        details=details or {},
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> bool:
    number = _as_float(value)
    return number is not None and math.isfinite(number)


def _positive_finite(value: Any) -> bool:
    number = _as_float(value)
    return number is not None and math.isfinite(number) and number > 0


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _required_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"reference {field_name} must be a boolean")
    return value


def _required_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ConfigError(f"reference {field_name} must be a SHA-256 digest")
    return value


def _required_sha256_mapping(
    value: Any,
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ConfigError(f"reference {field_name} must be a non-empty mapping")
    items: list[tuple[str, str]] = []
    for path, digest in value.items():
        if not isinstance(path, str) or not path:
            raise ConfigError(f"reference {field_name} has an invalid path")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ConfigError(f"reference {field_name} path must be relative")
        items.append((path, _required_sha256(digest, f"{field_name}.{path}")))
    return tuple(sorted(items))


def _required_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ConfigError(f"reference {field_name} must be a list of strings")
    return tuple(value)
