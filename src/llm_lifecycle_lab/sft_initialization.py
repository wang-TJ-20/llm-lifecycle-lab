"""Verify the frozen Base-v1 initialization boundary for SFT."""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.config import config_sha256, load_mapping, load_run_config
from llm_lifecycle_lab.contracts import (
    SCHEMA_VERSION,
    CheckpointMetadata,
    FileFingerprint,
    JsonContract,
    RunManifest,
    utc_now,
)
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.doctor.result import CheckResult, CheckStatus
from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError, ContractError
from llm_lifecycle_lab.model.native import (
    NativeModelConfig,
    NativeTransformer,
    load_native_model_config,
)
from llm_lifecycle_lab.tokenizer import NativeTokenizer

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PACKAGE_FILES = frozenset(
    {
        "LICENSE_MODEL",
        "config.json",
        "model.pt",
        "provenance.json",
        "tokenizer.json",
        "tokenizer_manifest.json",
    }
)
_EVIDENCE_FILES = frozenset(
    {
        "checkpoint_metadata.json",
        "final-test.json",
        "metrics.jsonl",
        "model_manifest.json",
        "resolved_config.yaml",
        "run_manifest.json",
        "tokenizer_manifest.json",
        "training_budget.json",
        "training_result.json",
    }
)


@dataclass(frozen=True, slots=True)
class SFTInitializationReport(JsonContract):
    gate_id: str
    root: str
    checks: tuple[CheckResult, ...]
    scope: str = "inputs-only"
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
class _Paths:
    base_package: Path
    source_evidence: Path
    source_pipeline: Path
    source_model_config: Path
    release_manifest: Path


@dataclass(frozen=True, slots=True)
class _Identity:
    model_route: str
    model_id: str
    architecture: str
    source_stage: str
    source_run_id: str
    source_run_profile: str
    source_seed: int
    source_git_commit: str
    source_sha256: str
    source_run_config_sha256: str
    source_pipeline_file_sha256: str
    source_model_config_sha256: str
    weights_sha256: str
    tokenizer_sha256: str
    tokenizer_manifest_sha256: str
    tokenizer_id: str
    chat_template_version: str
    parameter_count: int
    vocab_size: int
    max_sequence_length: int
    checkpoint_step: int
    license: str


@dataclass(frozen=True, slots=True)
class _Quality:
    expected_source_status: str
    expected_global_step: int
    expected_target_train_tokens: int
    minimum_target_token_coverage: float
    maximum_target_token_coverage: float
    maximum_final_to_baseline_loss_ratio: float
    maximum_final_to_best_loss_ratio: float
    require_final_language_improvement: bool
    expected_test_split: str


@dataclass(frozen=True, slots=True)
class _Policy:
    initialization_mode: str
    inherit_optimizer_state: bool
    initial_global_step: int
    require_clean_git: bool
    accepted_source_dirty: bool
    require_source_deviation_record: bool


@dataclass(frozen=True, slots=True)
class _SFTInitializationSpec:
    gate_id: str
    paths: _Paths
    package_artifacts: tuple[FileFingerprint, ...]
    evidence_artifacts: tuple[FileFingerprint, ...]
    identity: _Identity
    quality: _Quality
    policy: _Policy


def verify_sft_initialization(
    spec_path: str | Path,
    *,
    workdir: str | Path = ".",
    check_git: bool = False,
) -> SFTInitializationReport:
    """Verify Base-v1 inputs; preflight additionally requires a clean worktree."""

    root = Path(workdir).resolve()
    scope = "preflight" if check_git else "inputs-only"
    spec = _load_spec(Path(spec_path), root=root)
    checks: list[CheckResult] = [_policy_check(spec.policy)]

    checks.append(_configuration_check(spec))
    checks.append(
        _artifact_check(
            "base-artifacts",
            spec.paths.base_package,
            spec.package_artifacts,
            "canonical Base package matches every frozen artifact",
        )
    )
    checks.append(
        _artifact_check(
            "source-evidence-artifacts",
            spec.paths.source_evidence,
            spec.evidence_artifacts,
            "canonical Base evidence matches every frozen artifact",
        )
    )
    if _has_failures(checks):
        return _report(spec, root, scope, checks)

    checks.append(_release_manifest_check(spec))
    checks.append(_provenance_check(spec))
    checks.append(_evidence_contract_check(spec))
    checks.append(_quality_check(spec))
    if _has_failures(checks):
        return _report(spec, root, scope, checks)

    model_config, model_check = _model_tokenizer_check(spec)
    checks.append(model_check)
    if model_config is not None and not _has_failures(checks):
        checks.append(_model_loadability_check(spec, model_config))
    if check_git and not _has_failures(checks):
        checks.append(_clean_git_check(root, spec.policy.require_clean_git))
    return _report(spec, root, scope, checks)


def _report(
    spec: _SFTInitializationSpec,
    root: Path,
    scope: str,
    checks: list[CheckResult],
) -> SFTInitializationReport:
    return SFTInitializationReport(
        gate_id=spec.gate_id,
        root=str(root),
        checks=tuple(checks),
        scope=scope,
    )


def _policy_check(policy: _Policy) -> CheckResult:
    matches = (
        policy.initialization_mode == "weights-only-new-stage"
        and policy.inherit_optimizer_state is False
        and policy.initial_global_step == 0
        and policy.require_clean_git is True
        and policy.accepted_source_dirty is True
        and policy.require_source_deviation_record is True
    )
    return _check(
        "initialization-policy",
        matches,
        "SFT starts a clean stage from Base weights only",
        "SFT initialization policy would inherit state or weaken provenance",
        {
            "initialization_mode": policy.initialization_mode,
            "inherit_optimizer_state": policy.inherit_optimizer_state,
            "initial_global_step": policy.initial_global_step,
            "require_clean_git": policy.require_clean_git,
            "accepted_source_dirty": policy.accepted_source_dirty,
        },
    )


def _configuration_check(spec: _SFTInitializationSpec) -> CheckResult:
    identity = spec.identity
    details: dict[str, Any] = {}
    try:
        source = load_run_config(spec.paths.source_pipeline)
        resolved = load_run_config(spec.paths.source_evidence / "resolved_config.yaml")
        source_model = load_native_model_config(spec.paths.source_model_config)
        packaged_model = load_native_model_config(
            spec.paths.base_package / "config.json"
        )
        details = {
            "pipeline_file_sha256": sha256_file(spec.paths.source_pipeline),
            "resolved_config_sha256": config_sha256(source),
            "source_model_config_sha256": sha256_file(spec.paths.source_model_config),
        }
        matches = (
            details["pipeline_file_sha256"] == identity.source_pipeline_file_sha256
            and details["resolved_config_sha256"] == identity.source_run_config_sha256
            and config_sha256(resolved) == identity.source_run_config_sha256
            and details["source_model_config_sha256"]
            == identity.source_model_config_sha256
            and source == resolved
            and source.stage.value == identity.source_stage
            and source.model_route.value == identity.model_route
            and source.run_profile.value == identity.source_run_profile
            and source.seed == identity.source_seed
            and source.model.get("model_id") == identity.model_id
            and source_model == packaged_model
        )
    except (ConfigError, ArtifactError, OSError) as exc:
        matches = False
        details = {"error": str(exc)}
    return _check(
        "configuration-lock",
        matches,
        "source pipeline, resolved config and model config match Base-v1",
        "Base-v1 source or resolved configuration drifted",
        details,
    )


def _artifact_check(
    name: str,
    directory: Path,
    artifacts: tuple[FileFingerprint, ...],
    passed: str,
) -> CheckResult:
    mismatches: dict[str, Any] = {}
    for artifact in artifacts:
        path = directory / artifact.path
        actual_size: int | None = None
        actual_sha256: str | None = None
        if path.is_file() and not path.is_symlink():
            actual_size = path.stat().st_size
            actual_sha256 = sha256_file(path)
        if actual_size != artifact.size_bytes or actual_sha256 != artifact.sha256:
            mismatches[artifact.path] = {
                "expected_size_bytes": artifact.size_bytes,
                "actual_size_bytes": actual_size,
                "expected_sha256": artifact.sha256,
                "actual_sha256": actual_sha256,
            }
    return _check(
        name,
        not mismatches,
        passed,
        f"{name} differ from the frozen SFT initialization boundary",
        {"directory": str(directory), "mismatches": mismatches},
    )


def _release_manifest_check(spec: _SFTInitializationSpec) -> CheckResult:
    expected = {artifact.path: artifact.sha256 for artifact in spec.package_artifacts}
    actual: dict[str, str] = {}
    errors: list[str] = []
    try:
        lines = spec.paths.release_manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        lines = []
        errors.append(str(exc))
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if (
            len(parts) != 2
            or not _SHA256_PATTERN.fullmatch(parts[0])
            or parts[1] in actual
        ):
            errors.append(f"invalid SHA256SUMS entry: {line}")
            continue
        actual[parts[1]] = parts[0]
    mismatches = {
        path: {"expected": digest, "actual": actual.get(path)}
        for path, digest in expected.items()
        if actual.get(path) != digest
    }
    return _check(
        "release-manifest",
        not errors and not mismatches,
        "release manifest pins every training-critical Base artifact",
        "release manifest is malformed or disagrees with frozen Base artifacts",
        {"errors": errors, "mismatches": mismatches},
    )


def _provenance_check(spec: _SFTInitializationSpec) -> CheckResult:
    identity = spec.identity
    try:
        value = _load_json(
            spec.paths.base_package / "provenance.json", "Base provenance"
        )
        package = _mapping(value.get("package"))
        source = _mapping(value.get("source_run"))
        checkpoint = _mapping(value.get("checkpoint"))
        weights = _mapping(value.get("weights"))
        model = _mapping(value.get("model_config"))
        tokenizer = _mapping(value.get("tokenizer"))
        license_value = _mapping(package.get("license"))
        deviation = source.get("provenance_deviation")
        matches = (
            package.get("model_id") == identity.model_id
            and package.get("model_route") == identity.model_route
            and package.get("stage") == identity.source_stage
            and license_value.get("identifier") == identity.license
            and source.get("run_id") == identity.source_run_id
            and source.get("run_profile") == identity.source_run_profile
            and source.get("seed") == identity.source_seed
            and source.get("resolved_config_sha256")
            == identity.source_run_config_sha256
            and source.get("git_commit") == identity.source_git_commit
            and source.get("source_sha256") == identity.source_sha256
            and source.get("status") == spec.quality.expected_source_status
            and source.get("dirty") is spec.policy.accepted_source_dirty
            and (
                not spec.policy.require_source_deviation_record
                or isinstance(deviation, str)
                and bool(deviation.strip())
            )
            and checkpoint.get("global_step") == identity.checkpoint_step
            and checkpoint.get("config_sha256") == identity.source_run_config_sha256
            and checkpoint.get("tokenizer_sha256") == identity.tokenizer_sha256
            and weights.get("sha256") == identity.weights_sha256
            and weights.get("license") == identity.license
            and model.get("model_id") == identity.model_id
            and model.get("sha256")
            == _fingerprint(spec.package_artifacts, "config.json").sha256
            and tokenizer.get("tokenizer_id") == identity.tokenizer_id
            and tokenizer.get("vocab_size") == identity.vocab_size
            and tokenizer.get("chat_template_version") == identity.chat_template_version
            and tokenizer.get("content_sha256") == identity.tokenizer_sha256
            and tokenizer.get("license") == identity.license
        )
        details = {
            "source_run_id": source.get("run_id"),
            "source_dirty": source.get("dirty"),
            "source_deviation_recorded": isinstance(deviation, str)
            and bool(deviation.strip()),
            "checkpoint_step": checkpoint.get("global_step"),
        }
    except (ArtifactError, KeyError) as exc:
        matches = False
        details = {"error": str(exc)}
    return _check(
        "base-provenance",
        matches,
        "Base provenance matches the frozen source run and accepted deviation",
        "Base provenance does not match the frozen source run",
        details,
    )


def _evidence_contract_check(spec: _SFTInitializationSpec) -> CheckResult:
    identity = spec.identity
    quality = spec.quality
    evidence = spec.paths.source_evidence
    try:
        run = RunManifest.from_dict(
            _load_json(evidence / "run_manifest.json", "run manifest")
        )
        checkpoint = CheckpointMetadata.from_dict(
            _load_json(evidence / "checkpoint_metadata.json", "checkpoint metadata")
        )
        budget = _load_json(evidence / "training_budget.json", "training budget")
        result = _load_json(evidence / "training_result.json", "training result")
        model = _load_json(evidence / "model_manifest.json", "model manifest")
        model_metadata = _mapping(model.get("metadata"))
        matches = (
            run.run_id == identity.source_run_id
            and run.config_sha256 == identity.source_run_config_sha256
            and run.model_route.value == identity.model_route
            and run.run_profile.value == identity.source_run_profile
            and run.stage.value == identity.source_stage
            and run.status == quality.expected_source_status
            and checkpoint.run_id == identity.source_run_id
            and checkpoint.model_route.value == identity.model_route
            and checkpoint.stage.value == identity.source_stage
            and checkpoint.step == identity.checkpoint_step
            and checkpoint.config_sha256 == identity.source_run_config_sha256
            and checkpoint.tokenizer_sha256 == identity.tokenizer_sha256
            and budget.get("max_steps") == quality.expected_global_step
            and budget.get("target_train_tokens")
            == quality.expected_target_train_tokens
            and result.get("global_step") == quality.expected_global_step
            and model.get("config_sha256") == identity.source_model_config_sha256
            and model_metadata.get("model_route") == identity.model_route
            and model_metadata.get("model_id") == identity.model_id
            and model_metadata.get("architecture") == identity.architecture
            and model_metadata.get("parameter_count") == identity.parameter_count
            and model_metadata.get("tokenizer_revision") == identity.tokenizer_sha256
            and model_metadata.get("chat_template_version")
            == identity.chat_template_version
        )
        details = {
            "run_id": run.run_id,
            "status": run.status,
            "global_step": result.get("global_step"),
            "target_train_tokens": budget.get("target_train_tokens"),
        }
    except (ArtifactError, ContractError, KeyError, TypeError, ValueError) as exc:
        matches = False
        details = {"error": str(exc)}
    return _check(
        "source-evidence-contract",
        matches,
        "run, checkpoint, budget and model evidence agree on Base-v1 identity",
        "Base-v1 evidence files disagree on run or model identity",
        details,
    )


def _quality_check(spec: _SFTInitializationSpec) -> CheckResult:
    evidence = spec.paths.source_evidence
    quality = spec.quality
    try:
        metrics = _load_metrics(evidence / "metrics.jsonl")
        baseline_rows = [
            row
            for row in metrics
            if row.get("event") == "baseline" and row.get("step") == 0
        ]
        final_rows = [
            row for row in metrics if row.get("step") == quality.expected_global_step
        ]
        if len(baseline_rows) != 1 or len(final_rows) != 1:
            raise ArtifactError(
                "metrics require exactly one baseline and one final-step record"
            )
        baseline = baseline_rows[0]
        final = final_rows[0]
        result = _load_json(evidence / "training_result.json", "training result")
        test = _load_json(evidence / "final-test.json", "final test")
        baseline_loss = _number(baseline.get("eval_loss"))
        final_loss = _number(final.get("eval_loss"))
        best_loss = _number(result.get("best_eval_loss"))
        coverage = _number(result.get("target_token_coverage"))
        eval_losses = [
            value
            for row in metrics
            if (value := _number(row.get("eval_loss"))) is not None
            and row.get("step") != 0
        ]
        ratios_valid = (
            baseline_loss is not None
            and final_loss is not None
            and best_loss is not None
            and baseline_loss > 0
            and best_loss > 0
            and final_loss / baseline_loss
            <= quality.maximum_final_to_baseline_loss_ratio
            and final_loss / best_loss <= quality.maximum_final_to_best_loss_ratio
        )
        language_improved = all(
            _strictly_improved(baseline, final, key)
            for key in ("eval_en_loss", "eval_zh_loss")
        )
        test_valid = _valid_test_report(
            test,
            expected_step=quality.expected_global_step,
            expected_split=quality.expected_test_split,
        )
        matches = (
            result.get("global_step") == quality.expected_global_step
            and result.get("target_train_tokens")
            == quality.expected_target_train_tokens
            and coverage is not None
            and quality.minimum_target_token_coverage
            <= coverage
            <= quality.maximum_target_token_coverage
            and ratios_valid
            and (not quality.require_final_language_improvement or language_improved)
            and eval_losses
            and best_loss is not None
            and math.isclose(min(eval_losses), best_loss, rel_tol=1e-12, abs_tol=1e-12)
            and test_valid
        )
        details = {
            "baseline_eval_loss": baseline_loss,
            "final_eval_loss": final_loss,
            "best_eval_loss": best_loss,
            "target_token_coverage": coverage,
            "language_improved": language_improved,
            "test_eval_loss": test.get("eval_loss"),
        }
    except (ArtifactError, OSError, ValueError) as exc:
        matches = False
        details = {"error": str(exc)}
    return _check(
        "base-quality",
        bool(matches),
        "Base-v1 completion, bilingual improvement and held-out test are valid",
        "Base-v1 quality evidence does not satisfy the frozen SFT prerequisite",
        details,
    )


def _model_tokenizer_check(
    spec: _SFTInitializationSpec,
) -> tuple[NativeModelConfig | None, CheckResult]:
    identity = spec.identity
    try:
        model = load_native_model_config(spec.paths.base_package / "config.json")
        source_model = load_native_model_config(spec.paths.source_model_config)
        tokenizer = NativeTokenizer.from_directory(spec.paths.base_package)
        tokenizer_manifest_sha256 = sha256_file(
            spec.paths.base_package / "tokenizer_manifest.json"
        )
        matches = (
            model == source_model
            and model.model_id == identity.model_id
            and model.vocab_size == identity.vocab_size
            and model.max_sequence_length == identity.max_sequence_length
            and tokenizer.manifest.tokenizer_id == identity.tokenizer_id
            and tokenizer.manifest.content_sha256 == identity.tokenizer_sha256
            and tokenizer.manifest.chat_template_version
            == identity.chat_template_version
            and tokenizer_manifest_sha256 == identity.tokenizer_manifest_sha256
            and tokenizer.vocab_size == model.vocab_size
        )
        details = {
            "model_id": model.model_id,
            "vocab_size": model.vocab_size,
            "max_sequence_length": model.max_sequence_length,
            "tokenizer_id": tokenizer.manifest.tokenizer_id,
        }
        loaded = model
    except (ArtifactError, ConfigError, OSError) as exc:
        matches = False
        details = {"error": str(exc)}
        loaded = None
    return loaded, _check(
        "model-tokenizer-contract",
        matches,
        "model config and tokenizer are loadable and mutually compatible",
        "model config or tokenizer violates the frozen Base contract",
        details,
    )


def _model_loadability_check(
    spec: _SFTInitializationSpec,
    config: NativeModelConfig,
) -> CheckResult:
    try:
        parameter_count = _load_model_parameter_count(spec.paths.base_package, config)
        matches = parameter_count == spec.identity.parameter_count
        details: dict[str, Any] = {
            "parameter_count": parameter_count,
            "expected_parameter_count": spec.identity.parameter_count,
        }
    except (ArtifactError, ConfigError, RuntimeError, OSError, ValueError) as exc:
        matches = False
        details = {"error": str(exc)}
    return _check(
        "model-loadability",
        matches,
        "Base weights load strictly into the expected Native model",
        "Base weights do not load strictly or parameter count changed",
        details,
    )


def _load_model_parameter_count(
    package: Path,
    config: NativeModelConfig,
) -> int:
    model = NativeTransformer(config)
    model.load(package)
    return model.parameter_count


def _clean_git_check(root: Path, required: bool) -> CheckResult:
    commit = _run_git(root, "rev-parse", "--verify", "HEAD")
    status = _run_git(root, "status", "--porcelain=v1")
    clean = (
        not required
        or commit is not None
        and _COMMIT_PATTERN.fullmatch(commit) is not None
        and status == ""
    )
    return _check(
        "clean-worktree",
        clean,
        "current SFT worktree is clean and has a valid commit",
        "commit SFT inputs and code before starting a run",
        {
            "git_commit": commit,
            "git_dirty": None if status is None else bool(status),
            "status_entries": None if status is None else len(status.splitlines()),
        },
    )


def _load_spec(path: Path, *, root: Path) -> _SFTInitializationSpec:
    value = load_mapping(path)
    _exact_keys(
        value,
        {
            "schema_version",
            "gate_id",
            "paths",
            "package_artifacts",
            "evidence_artifacts",
            "identity",
            "quality",
            "policy",
        },
        "SFT initialization spec",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ConfigError("SFT initialization spec has unsupported schema_version")
    gate_id = _text(value["gate_id"], "gate_id")
    paths = _load_paths(value["paths"], root)
    package = _load_fingerprints(
        value["package_artifacts"], _PACKAGE_FILES, "package_artifacts"
    )
    evidence = _load_fingerprints(
        value["evidence_artifacts"], _EVIDENCE_FILES, "evidence_artifacts"
    )
    identity = _load_identity(value["identity"])
    quality = _load_quality(value["quality"])
    policy = _load_policy(value["policy"])
    return _SFTInitializationSpec(
        gate_id=gate_id,
        paths=paths,
        package_artifacts=package,
        evidence_artifacts=evidence,
        identity=identity,
        quality=quality,
        policy=policy,
    )


def _load_paths(value: Any, root: Path) -> _Paths:
    mapping = _required_mapping(value, "paths")
    keys = {
        "base_package",
        "source_evidence",
        "source_pipeline",
        "source_model_config",
        "release_manifest",
    }
    _exact_keys(mapping, keys, "paths")
    resolved = {
        key: _root_path(root, _text(mapping[key], f"paths.{key}")) for key in keys
    }
    return _Paths(**resolved)


def _load_fingerprints(
    value: Any,
    expected_paths: frozenset[str],
    name: str,
) -> tuple[FileFingerprint, ...]:
    mapping = _required_mapping(value, name)
    _exact_keys(mapping, set(expected_paths), name)
    artifacts: list[FileFingerprint] = []
    for path in sorted(mapping):
        item = _required_mapping(mapping[path], f"{name}.{path}")
        _exact_keys(item, {"sha256", "size_bytes"}, f"{name}.{path}")
        _relative_path(path, f"{name}.{path}")
        try:
            artifacts.append(
                FileFingerprint(
                    path=path,
                    sha256=_sha256(item["sha256"], f"{name}.{path}.sha256"),
                    size_bytes=_integer(
                        item["size_bytes"], f"{name}.{path}.size_bytes", minimum=0
                    ),
                )
            )
        except ContractError as exc:
            raise ConfigError(f"invalid {name}.{path}: {exc}") from exc
    return tuple(artifacts)


def _load_identity(value: Any) -> _Identity:
    mapping = _required_mapping(value, "identity")
    keys = {
        "model_route",
        "model_id",
        "architecture",
        "source_stage",
        "source_run_id",
        "source_run_profile",
        "source_seed",
        "source_git_commit",
        "source_sha256",
        "source_run_config_sha256",
        "source_pipeline_file_sha256",
        "source_model_config_sha256",
        "weights_sha256",
        "tokenizer_sha256",
        "tokenizer_manifest_sha256",
        "tokenizer_id",
        "chat_template_version",
        "parameter_count",
        "vocab_size",
        "max_sequence_length",
        "checkpoint_step",
        "license",
    }
    _exact_keys(mapping, keys, "identity")
    return _Identity(
        model_route=_text(mapping["model_route"], "identity.model_route"),
        model_id=_text(mapping["model_id"], "identity.model_id"),
        architecture=_text(mapping["architecture"], "identity.architecture"),
        source_stage=_text(mapping["source_stage"], "identity.source_stage"),
        source_run_id=_text(mapping["source_run_id"], "identity.source_run_id"),
        source_run_profile=_text(
            mapping["source_run_profile"], "identity.source_run_profile"
        ),
        source_seed=_integer(mapping["source_seed"], "identity.source_seed", minimum=0),
        source_git_commit=_commit(
            mapping["source_git_commit"], "identity.source_git_commit"
        ),
        source_sha256=_sha256(mapping["source_sha256"], "identity.source_sha256"),
        source_run_config_sha256=_sha256(
            mapping["source_run_config_sha256"],
            "identity.source_run_config_sha256",
        ),
        source_pipeline_file_sha256=_sha256(
            mapping["source_pipeline_file_sha256"],
            "identity.source_pipeline_file_sha256",
        ),
        source_model_config_sha256=_sha256(
            mapping["source_model_config_sha256"],
            "identity.source_model_config_sha256",
        ),
        weights_sha256=_sha256(mapping["weights_sha256"], "identity.weights_sha256"),
        tokenizer_sha256=_sha256(
            mapping["tokenizer_sha256"], "identity.tokenizer_sha256"
        ),
        tokenizer_manifest_sha256=_sha256(
            mapping["tokenizer_manifest_sha256"],
            "identity.tokenizer_manifest_sha256",
        ),
        tokenizer_id=_text(mapping["tokenizer_id"], "identity.tokenizer_id"),
        chat_template_version=_text(
            mapping["chat_template_version"], "identity.chat_template_version"
        ),
        parameter_count=_integer(
            mapping["parameter_count"], "identity.parameter_count", minimum=1
        ),
        vocab_size=_integer(mapping["vocab_size"], "identity.vocab_size", minimum=1),
        max_sequence_length=_integer(
            mapping["max_sequence_length"],
            "identity.max_sequence_length",
            minimum=2,
        ),
        checkpoint_step=_integer(
            mapping["checkpoint_step"], "identity.checkpoint_step", minimum=1
        ),
        license=_text(mapping["license"], "identity.license"),
    )


def _load_quality(value: Any) -> _Quality:
    mapping = _required_mapping(value, "quality")
    keys = {
        "expected_source_status",
        "expected_global_step",
        "expected_target_train_tokens",
        "minimum_target_token_coverage",
        "maximum_target_token_coverage",
        "maximum_final_to_baseline_loss_ratio",
        "maximum_final_to_best_loss_ratio",
        "require_final_language_improvement",
        "expected_test_split",
    }
    _exact_keys(mapping, keys, "quality")
    minimum_coverage = _positive_number(
        mapping["minimum_target_token_coverage"],
        "quality.minimum_target_token_coverage",
    )
    maximum_coverage = _positive_number(
        mapping["maximum_target_token_coverage"],
        "quality.maximum_target_token_coverage",
    )
    if maximum_coverage < minimum_coverage:
        raise ConfigError("quality token coverage bounds are inverted")
    return _Quality(
        expected_source_status=_text(
            mapping["expected_source_status"], "quality.expected_source_status"
        ),
        expected_global_step=_integer(
            mapping["expected_global_step"],
            "quality.expected_global_step",
            minimum=1,
        ),
        expected_target_train_tokens=_integer(
            mapping["expected_target_train_tokens"],
            "quality.expected_target_train_tokens",
            minimum=1,
        ),
        minimum_target_token_coverage=minimum_coverage,
        maximum_target_token_coverage=maximum_coverage,
        maximum_final_to_baseline_loss_ratio=_positive_number(
            mapping["maximum_final_to_baseline_loss_ratio"],
            "quality.maximum_final_to_baseline_loss_ratio",
        ),
        maximum_final_to_best_loss_ratio=_positive_number(
            mapping["maximum_final_to_best_loss_ratio"],
            "quality.maximum_final_to_best_loss_ratio",
        ),
        require_final_language_improvement=_boolean(
            mapping["require_final_language_improvement"],
            "quality.require_final_language_improvement",
        ),
        expected_test_split=_text(
            mapping["expected_test_split"], "quality.expected_test_split"
        ),
    )


def _load_policy(value: Any) -> _Policy:
    mapping = _required_mapping(value, "policy")
    keys = {
        "initialization_mode",
        "inherit_optimizer_state",
        "initial_global_step",
        "require_clean_git",
        "accepted_source_dirty",
        "require_source_deviation_record",
    }
    _exact_keys(mapping, keys, "policy")
    return _Policy(
        initialization_mode=_text(
            mapping["initialization_mode"], "policy.initialization_mode"
        ),
        inherit_optimizer_state=_boolean(
            mapping["inherit_optimizer_state"], "policy.inherit_optimizer_state"
        ),
        initial_global_step=_integer(
            mapping["initial_global_step"],
            "policy.initial_global_step",
            minimum=0,
        ),
        require_clean_git=_boolean(
            mapping["require_clean_git"], "policy.require_clean_git"
        ),
        accepted_source_dirty=_boolean(
            mapping["accepted_source_dirty"], "policy.accepted_source_dirty"
        ),
        require_source_deviation_record=_boolean(
            mapping["require_source_deviation_record"],
            "policy.require_source_deviation_record",
        ),
    )


def _valid_test_report(
    value: Mapping[str, Any],
    *,
    expected_step: int,
    expected_split: str,
) -> bool:
    keys = (
        "eval_loss",
        "eval_perplexity",
        "eval_en_loss",
        "eval_en_perplexity",
        "eval_en_tokens",
        "eval_zh_loss",
        "eval_zh_perplexity",
        "eval_zh_tokens",
        "eval_tokens",
        "eval_bits_per_byte",
    )
    if (
        value.get("checkpoint_step") != expected_step
        or value.get("split") != expected_split
        or any(
            (number := _number(value.get(key))) is None or number <= 0 for key in keys
        )
    ):
        return False
    total = _number(value.get("eval_tokens"))
    en_tokens = _number(value.get("eval_en_tokens"))
    zh_tokens = _number(value.get("eval_zh_tokens"))
    loss = _number(value.get("eval_loss"))
    en_loss = _number(value.get("eval_en_loss"))
    zh_loss = _number(value.get("eval_zh_loss"))
    assert None not in (total, en_tokens, zh_tokens, loss, en_loss, zh_loss)
    assert total is not None
    assert en_tokens is not None
    assert zh_tokens is not None
    assert loss is not None
    assert en_loss is not None
    assert zh_loss is not None
    weighted = (en_loss * en_tokens + zh_loss * zh_tokens) / total
    return math.isclose(total, en_tokens + zh_tokens, abs_tol=1e-6) and math.isclose(
        loss, weighted, rel_tol=1e-6, abs_tol=1e-6
    )


def _strictly_improved(
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
    key: str,
) -> bool:
    before = _number(baseline.get(key))
    after = _number(final.get(key))
    return before is not None and after is not None and 0 < after < before


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must contain a JSON object: {path}")
    return value


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ArtifactError(
                        f"metrics line {line_number} is not a JSON object"
                    )
                values.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read metrics {path}: {exc}") from exc
    if not values:
        raise ArtifactError(f"metrics file is empty: {path}")
    return values


def _fingerprint(
    artifacts: tuple[FileFingerprint, ...],
    path: str,
) -> FileFingerprint:
    for artifact in artifacts:
        if artifact.path == path:
            return artifact
    raise KeyError(path)


def _has_failures(checks: list[CheckResult]) -> bool:
    return any(check.status is CheckStatus.FAIL for check in checks)


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


def _run_git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if unknown:
            parts.append("unknown " + ", ".join(unknown))
        raise ConfigError(f"{name} fields are invalid: {'; '.join(parts)}")


def _root_path(root: Path, value: str) -> Path:
    relative = _relative_path(value, "frozen path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConfigError("frozen path resolves outside the repository") from exc
    return path


def _relative_path(value: str, name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ConfigError(f"{name} must remain inside the repository")
    return path


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, name: str) -> float:
    number = _number(value)
    if number is None or number <= 0:
        raise ConfigError(f"{name} must be finite and positive")
    return number


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ConfigError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _COMMIT_PATTERN.fullmatch(value):
        raise ConfigError(f"{name} must be a 40-character commit SHA")
    return value
