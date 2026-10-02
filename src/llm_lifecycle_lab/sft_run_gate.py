"""Verify completed Native-60M SFT runs before dev acceptance or sealed test."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.config import config_sha256, load_mapping, load_run_config
from llm_lifecycle_lab.contracts import (
    CheckpointMetadata,
    EvaluationReport,
    JsonContract,
    ModelRoute,
    RunManifest,
    RunProfile,
    Stage,
    utc_now,
)
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.doctor.result import CheckResult, CheckStatus
from llm_lifecycle_lab.evaluation.suite import read_json
from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError, ContractError

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SFTRunGateReport(JsonContract):
    gate_id: str
    run_path: str
    checks: tuple[CheckResult, ...]
    scope: str
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
class _Locks:
    pipeline_config: Path
    pipeline_file_sha256: str
    pipeline_config_sha256: str
    base_gate: Path
    base_gate_sha256: str
    data_gate: Path
    data_gate_sha256: str
    evaluation_suite: Path
    evaluation_suite_file_sha256: str
    evaluation_suite_sha256: str


@dataclass(frozen=True, slots=True)
class _Identity:
    model_route: ModelRoute
    run_profile: RunProfile
    stage: Stage
    parent_run_id: str
    parent_checkpoint_step: int
    parent_weights_sha256: str
    parent_model_config_sha256: str
    tokenizer_sha256: str
    data_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _Training:
    budget_mode: str
    global_step: int
    target_train_tokens: int
    expected_tokens_seen: int
    minimum_target_token_coverage: float
    maximum_target_token_coverage: float
    required_training_metric_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Evaluation:
    dev_suite: str
    dev_examples: int
    test_suite: str
    test_examples: int
    required_metric_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Runtime:
    platform_system: str
    accelerator_type: str
    require_clean_source: bool


@dataclass(frozen=True, slots=True)
class _Spec:
    gate_id: str
    locks: _Locks
    identity: _Identity
    training: _Training
    evaluation: _Evaluation
    runtime: _Runtime


def verify_sft_run(
    spec_path: str | Path,
    run_path: str | Path,
    *,
    workdir: str | Path = ".",
    sealed_test: bool = False,
) -> SFTRunGateReport:
    """Verify a formal completed run and optionally require its sealed test."""

    root = Path(workdir).resolve()
    spec = _load_spec(Path(spec_path), root)
    run = _resolve_path(run_path, root)
    scope = "sealed-test" if sealed_test else "development"
    checks = [_execution_lock_check(spec, root)]
    if _has_failures(checks):
        return _report(spec, run, scope, checks)

    required = (
        "run_manifest.json",
        "resolved_config.yaml",
        "initialization.json",
        "preflight.json",
        "sft_data_summary.json",
        "training_budget.json",
        "training_result.json",
        "runtime_environment.json",
        "metrics.jsonl",
        "latest_checkpoint.json",
        "model_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_manifest.json",
    )
    missing = [name for name in required if not (run / name).is_file()]
    checks.append(
        _check(
            "required-artifacts",
            not missing,
            "all required SFT run artifacts are present",
            "missing SFT run artifacts: " + ", ".join(missing),
            {"missing": missing},
        )
    )
    if missing:
        return _report(spec, run, scope, checks)

    manifest = _load_contract(
        run / "run_manifest.json", RunManifest.from_dict, "run manifest"
    )
    resolved = load_run_config(run / "resolved_config.yaml")
    checks.append(_run_contract_check(spec, manifest, resolved))

    preflight = _load_json(run / "preflight.json", "preflight")
    checks.append(_preflight_check(spec, preflight))
    initialization = _load_json(run / "initialization.json", "initialization")
    checks.append(_initialization_check(spec, initialization))

    budget = _load_json(run / "training_budget.json", "training budget")
    result = _load_json(run / "training_result.json", "training result")
    checks.append(_training_budget_check(spec, budget, result))
    checks.append(_training_result_check(spec, result))

    metrics = _load_metrics(run / "metrics.jsonl")
    checks.append(_metrics_check(spec, metrics, result))
    checks.append(_checkpoint_check(spec, run, manifest, result))

    checks.append(
        _evaluation_check(
            spec,
            run,
            manifest.run_id,
            suite=spec.evaluation.dev_suite,
            examples=spec.evaluation.dev_examples,
            name="dev-evaluation",
        )
    )
    if sealed_test:
        checks.append(
            _evaluation_check(
                spec,
                run,
                manifest.run_id,
                suite=spec.evaluation.test_suite,
                examples=spec.evaluation.test_examples,
                name="sealed-test-evaluation",
            )
        )

    runtime = _load_json(run / "runtime_environment.json", "runtime environment")
    checks.append(_runtime_check(spec, runtime))
    return _report(spec, run, scope, checks)


def _report(
    spec: _Spec,
    run: Path,
    scope: str,
    checks: list[CheckResult],
) -> SFTRunGateReport:
    return SFTRunGateReport(
        gate_id=spec.gate_id,
        run_path=str(run),
        checks=tuple(checks),
        scope=scope,
    )


def _execution_lock_check(spec: _Spec, root: Path) -> CheckResult:
    locks = spec.locks
    files = {
        "pipeline_config": (locks.pipeline_config, locks.pipeline_file_sha256),
        "base_gate": (locks.base_gate, locks.base_gate_sha256),
        "data_gate": (locks.data_gate, locks.data_gate_sha256),
        "evaluation_suite": (
            locks.evaluation_suite,
            locks.evaluation_suite_file_sha256,
        ),
    }
    actual = {
        name: (
            sha256_file(path_and_hash[0])
            if path_and_hash[0].is_file() and not path_and_hash[0].is_symlink()
            else None
        )
        for name, path_and_hash in files.items()
    }
    config = load_run_config(locks.pipeline_config)
    declared = {
        "base_gate": _config_path(config.data, "base_gate", root),
        "data_gate": _config_path(config.data, "data_gate", root),
        "evaluation_suite": _config_path(config.data, "evaluation_suite", root),
    }
    matches = (
        all(actual[name] == path_and_hash[1] for name, path_and_hash in files.items())
        and config_sha256(config) == locks.pipeline_config_sha256
        and config.model_route is spec.identity.model_route
        and config.run_profile is spec.identity.run_profile
        and config.stage is spec.identity.stage
        and all(
            declared[name].resolve() == getattr(locks, name).resolve()
            for name in declared
        )
        and config.data.get("evaluation_suite_sha256")
        == locks.evaluation_suite_file_sha256
    )
    return _check(
        "execution-lock",
        matches,
        "formal pipeline, gates and evaluation suite match the frozen run spec",
        "formal SFT execution inputs differ from the frozen run spec",
        {
            "actual_file_sha256": actual,
            "pipeline_config_sha256": config_sha256(config),
            "expected_pipeline_config_sha256": locks.pipeline_config_sha256,
        },
    )


def _run_contract_check(
    spec: _Spec,
    manifest: RunManifest,
    resolved: Any,
) -> CheckResult:
    matches = (
        manifest.status == "completed"
        and manifest.model_route is spec.identity.model_route
        and manifest.run_profile is spec.identity.run_profile
        and manifest.stage is spec.identity.stage
        and manifest.config_sha256 == spec.locks.pipeline_config_sha256
        and config_sha256(resolved) == spec.locks.pipeline_config_sha256
    )
    return _check(
        "run-contract",
        matches,
        "completed run identity and resolved configuration are frozen",
        "run status, stage, profile, route or configuration hash is invalid",
        {
            "run_id": manifest.run_id,
            "status": manifest.status,
            "config_sha256": manifest.config_sha256,
        },
    )


def _preflight_check(
    spec: _Spec,
    value: Mapping[str, Any],
) -> CheckResult:
    base = _optional_mapping(value.get("base_gate"))
    data = _optional_mapping(value.get("data_gate"))
    suite = _optional_mapping(value.get("evaluation_suite"))
    suite_path = suite.get("path")
    suite_matches = (
        isinstance(suite_path, str)
        and Path(suite_path).resolve() == spec.locks.evaluation_suite.resolve()
    )
    matches = (
        value.get("scope") == "preflight"
        and value.get("git_checked") is True
        and base.get("gate_id") == "native-60m-sft-v1-base-init"
        and base.get("ok") is True
        and data.get("gate_id") == "native-60m-sft-v1-data"
        and data.get("scope") == "preflight"
        and data.get("ok") is True
        and suite.get("sha256") == spec.locks.evaluation_suite_file_sha256
        and suite_matches
    )
    return _check(
        "preflight",
        matches,
        "recorded Base, data, suite and clean-worktree preflight passed",
        "run does not contain a passing formal SFT preflight",
        {
            "scope": value.get("scope"),
            "git_checked": value.get("git_checked"),
            "base_ok": base.get("ok"),
            "data_ok": data.get("ok"),
        },
    )


def _initialization_check(
    spec: _Spec,
    value: Mapping[str, Any],
) -> CheckResult:
    parent = _optional_mapping(value.get("parent_checkpoint"))
    sampling = _optional_mapping(value.get("sampling"))
    matches = (
        value.get("mode") == "weights-only-new-stage"
        and value.get("optimizer_inherited") is False
        and _as_int(value.get("initial_step")) == 0
        and parent.get("run_id") == spec.identity.parent_run_id
        and _as_int(parent.get("step")) == spec.identity.parent_checkpoint_step
        and parent.get("model_route") == ModelRoute.NATIVE.value
        and parent.get("stage") == Stage.PRETRAIN.value
        and parent.get("tokenizer_sha256") == spec.identity.tokenizer_sha256
        and value.get("parent_weights_sha256") == spec.identity.parent_weights_sha256
        and value.get("parent_model_config_sha256")
        == spec.identity.parent_model_config_sha256
        and value.get("data_manifest_sha256") == spec.identity.data_manifest_sha256
        and value.get("evaluation_suite_sha256") == spec.locks.evaluation_suite_sha256
        and sampling.get("strategy") == "supervised-token-quota"
    )
    return _check(
        "initialization",
        matches,
        "SFT starts at step 0 from the frozen Base weights and data",
        "SFT initialization, parent, tokenizer, data or suite identity differs",
    )


def _training_budget_check(
    spec: _Spec,
    budget: Mapping[str, Any],
    result: Mapping[str, Any],
) -> CheckResult:
    expected = spec.training
    matches = (
        budget.get("mode") == expected.budget_mode
        and _as_int(budget.get("max_steps")) == expected.global_step
        and _as_int(budget.get("target_train_tokens")) == expected.target_train_tokens
        and _as_int(budget.get("requested_max_train_tokens"))
        == expected.target_train_tokens
        and budget.get("requested_max_steps") is None
        and budget.get("requested_num_epochs") is None
        and _as_int(result.get("global_step")) == expected.global_step
        and _as_int(result.get("target_train_tokens")) == expected.target_train_tokens
    )
    return _check(
        "training-budget",
        matches,
        "resolved optimizer steps and 8M supervised-token target match",
        "training budget, completed step or token target differs",
        {
            "mode": budget.get("mode"),
            "global_step": result.get("global_step"),
            "target_train_tokens": result.get("target_train_tokens"),
        },
    )


def _training_result_check(
    spec: _Spec,
    result: Mapping[str, Any],
) -> CheckResult:
    coverage = _as_float(result.get("target_token_coverage"))
    tokens_seen = _as_int(result.get("tokens_seen"))
    target = spec.training.target_train_tokens
    ratio_matches = (
        coverage is not None
        and tokens_seen is not None
        and math.isclose(coverage, tokens_seen / target, rel_tol=0.0, abs_tol=1e-12)
    )
    matches = (
        ratio_matches
        and tokens_seen == spec.training.expected_tokens_seen
        and spec.training.minimum_target_token_coverage
        <= coverage
        <= spec.training.maximum_target_token_coverage
        and _positive_finite(result.get("elapsed_seconds"))
        and _positive_finite(result.get("final_loss"))
        and _positive_finite(result.get("best_eval_loss"))
    )
    return _check(
        "training-result",
        matches,
        "training result is finite and token coverage is within bounds",
        "training result is non-finite or token coverage is invalid",
        {
            "tokens_seen": tokens_seen,
            "expected_tokens_seen": spec.training.expected_tokens_seen,
            "target_token_coverage": coverage,
            "minimum": spec.training.minimum_target_token_coverage,
            "maximum": spec.training.maximum_target_token_coverage,
        },
    )


def _metrics_check(
    spec: _Spec,
    rows: list[dict[str, Any]],
    result: Mapping[str, Any],
) -> CheckResult:
    baseline = next((row for row in rows if row.get("event") == "baseline"), {})
    final = next(
        (
            row
            for row in reversed(rows)
            if _as_int(row.get("step")) == spec.training.global_step
        ),
        {},
    )
    missing = [
        key
        for key in spec.training.required_training_metric_keys
        if not _finite(final.get(key))
    ]
    loss_keys = ("eval_loss", "eval_en_loss", "eval_zh_loss")
    improved = all(
        _finite(baseline.get(key))
        and _finite(final.get(key))
        and float(final[key]) < float(baseline[key])
        for key in loss_keys
    )
    evaluated = [row for row in rows if _finite(row.get("eval_loss"))]
    best = min((float(row["eval_loss"]) for row in evaluated), default=math.nan)
    recorded_best = _as_float(result.get("best_eval_loss"))
    matches = (
        baseline.get("event") == "baseline"
        and _as_int(baseline.get("step")) == 0
        and not missing
        and improved
        and _language_loss_is_consistent(baseline)
        and _language_loss_is_consistent(final)
        and recorded_best is not None
        and math.isclose(recorded_best, best, rel_tol=0.0, abs_tol=1e-9)
        and _finite(result.get("final_loss"))
        and math.isclose(
            float(result["final_loss"]),
            float(final.get("train_loss", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    return _check(
        "development-metrics",
        matches,
        "final dev aggregate, English and Chinese losses improve from baseline",
        "dev metrics are missing, inconsistent, non-finite or did not all improve",
        {
            "missing_metric_keys": missing,
            "baseline": {key: baseline.get(key) for key in loss_keys},
            "final": {key: final.get(key) for key in loss_keys},
        },
    )


def _checkpoint_check(
    spec: _Spec,
    run: Path,
    manifest: RunManifest,
    result: Mapping[str, Any],
) -> CheckResult:
    latest = _load_json(run / "latest_checkpoint.json", "latest checkpoint")
    checkpoint = _safe_child(run / "checkpoints", latest.get("path"), run)
    result_checkpoint = _safe_child(
        run / "checkpoints",
        result.get("final_checkpoint"),
        run,
    )
    complete = False
    if checkpoint is not None:
        required = (
            checkpoint / "checkpoint_metadata.json",
            checkpoint / "trainer_state.json",
            checkpoint / "optimizer_state.pt",
            checkpoint / "model/model.pt",
            checkpoint / "model/config.json",
        )
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
            metadata = _load_contract(
                checkpoint / "checkpoint_metadata.json",
                CheckpointMetadata.from_dict,
                "checkpoint metadata",
            )
            trainer = _load_json(checkpoint / "trainer_state.json", "trainer state")
            complete = (
                metadata.run_id == manifest.run_id
                and metadata.step == spec.training.global_step
                and metadata.model_route is spec.identity.model_route
                and metadata.stage is spec.identity.stage
                and metadata.config_sha256 == spec.locks.pipeline_config_sha256
                and metadata.tokenizer_sha256 == spec.identity.tokenizer_sha256
                and _as_int(latest.get("step")) == spec.training.global_step
                and _as_int(trainer.get("global_step")) == spec.training.global_step
                and _as_int(trainer.get("tokens_seen"))
                == _as_int(result.get("tokens_seen"))
                and result_checkpoint == checkpoint
                and sha256_file(checkpoint / "model/config.json")
                == spec.identity.parent_model_config_sha256
                and sha256_file(run / "tokenizer/tokenizer.json")
                == spec.identity.tokenizer_sha256
            )
    return _check(
        "final-checkpoint",
        complete,
        "final model, optimizer, trainer state and metadata are complete",
        "final checkpoint is missing, unsafe or incompatible",
    )


def _evaluation_check(
    spec: _Spec,
    run: Path,
    run_id: str,
    *,
    suite: str,
    examples: int,
    name: str,
) -> CheckResult:
    path = run / "evaluations" / f"{suite}-step-{spec.training.global_step:08d}.json"
    valid = False
    details: dict[str, Any] = {"path": str(path)}
    if path.is_file():
        raw = _load_json(path, f"{suite} evaluation")
        report = EvaluationReport(
            run_id=str(raw.get("run_id", "")),
            suite=str(raw.get("suite", "")),
            metrics=_optional_mapping(raw.get("metrics")),
            sample_count=_as_int(raw.get("sample_count")) or 0,
            created_at=str(raw.get("created_at", "")),
            schema_version=str(raw.get("schema_version", "")),
        )
        metrics = report.metrics
        missing = [
            key
            for key in spec.evaluation.required_metric_keys
            if not _finite(metrics.get(key))
        ]
        valid = (
            report.run_id == run_id
            and report.suite == suite
            and report.sample_count == examples
            and raw.get("schema_version") == "1.0"
            and not missing
            and _language_loss_is_consistent(metrics)
            and all(
                float(metrics[key]) > 0
                for key in (
                    "eval_loss",
                    "eval_perplexity",
                    "eval_en_loss",
                    "eval_en_perplexity",
                    "eval_zh_loss",
                    "eval_zh_perplexity",
                )
            )
        )
        details.update(
            {
                "sample_count": report.sample_count,
                "expected_sample_count": examples,
                "missing_metric_keys": missing,
            }
        )
    return _check(
        name,
        valid,
        f"complete frozen {suite} report is present and valid",
        f"complete frozen {suite} report is missing or invalid",
        details,
    )


def _runtime_check(spec: _Spec, value: Mapping[str, Any]) -> CheckResult:
    platform = _optional_mapping(value.get("platform"))
    accelerator = _optional_mapping(value.get("accelerator"))
    code = _optional_mapping(value.get("code"))
    matches = (
        platform.get("system") == spec.runtime.platform_system
        and accelerator.get("type") == spec.runtime.accelerator_type
        and (
            not spec.runtime.require_clean_source
            or (code.get("dirty") is False and _as_int(code.get("status_entries")) == 0)
        )
    )
    return _check(
        "runtime",
        matches,
        "run was created on the required accelerator from a clean source tree",
        "runtime platform, accelerator or source cleanliness is invalid",
        {
            "platform_system": platform.get("system"),
            "accelerator_type": accelerator.get("type"),
            "dirty": code.get("dirty"),
            "status_entries": code.get("status_entries"),
        },
    )


def _load_spec(path: Path, root: Path) -> _Spec:
    resolved = _resolve_path(path, root)
    value = load_mapping(resolved)
    _strict(
        value,
        {
            "schema_version",
            "gate_id",
            "locks",
            "identity",
            "training",
            "evaluation",
            "runtime",
        },
        "run gate",
    )
    if value.get("schema_version") != "1.0":
        raise ConfigError("unsupported SFT run gate schema_version")
    locks = _require_mapping(value.get("locks"), "locks")
    identity = _require_mapping(value.get("identity"), "identity")
    training = _require_mapping(value.get("training"), "training")
    evaluation = _require_mapping(value.get("evaluation"), "evaluation")
    runtime = _require_mapping(value.get("runtime"), "runtime")
    _strict(
        locks,
        {
            "pipeline_config",
            "pipeline_file_sha256",
            "pipeline_config_sha256",
            "base_gate",
            "base_gate_sha256",
            "data_gate",
            "data_gate_sha256",
            "evaluation_suite",
            "evaluation_suite_file_sha256",
            "evaluation_suite_sha256",
        },
        "locks",
    )
    _strict(
        identity,
        {
            "model_route",
            "run_profile",
            "stage",
            "parent_run_id",
            "parent_checkpoint_step",
            "parent_weights_sha256",
            "parent_model_config_sha256",
            "tokenizer_sha256",
            "data_manifest_sha256",
        },
        "identity",
    )
    _strict(
        training,
        {
            "budget_mode",
            "global_step",
            "target_train_tokens",
            "expected_tokens_seen",
            "minimum_target_token_coverage",
            "maximum_target_token_coverage",
            "required_training_metric_keys",
        },
        "training",
    )
    _strict(
        evaluation,
        {
            "dev_suite",
            "dev_examples",
            "test_suite",
            "test_examples",
            "required_metric_keys",
        },
        "evaluation",
    )
    _strict(
        runtime,
        {"platform_system", "accelerator_type", "require_clean_source"},
        "runtime",
    )
    spec = _Spec(
        gate_id=_text(value.get("gate_id"), "gate_id"),
        locks=_Locks(
            pipeline_config=_spec_path(locks, "pipeline_config", root),
            pipeline_file_sha256=_sha(
                locks.get("pipeline_file_sha256"), "pipeline_file_sha256"
            ),
            pipeline_config_sha256=_sha(
                locks.get("pipeline_config_sha256"), "pipeline_config_sha256"
            ),
            base_gate=_spec_path(locks, "base_gate", root),
            base_gate_sha256=_sha(locks.get("base_gate_sha256"), "base_gate_sha256"),
            data_gate=_spec_path(locks, "data_gate", root),
            data_gate_sha256=_sha(locks.get("data_gate_sha256"), "data_gate_sha256"),
            evaluation_suite=_spec_path(locks, "evaluation_suite", root),
            evaluation_suite_file_sha256=_sha(
                locks.get("evaluation_suite_file_sha256"),
                "evaluation_suite_file_sha256",
            ),
            evaluation_suite_sha256=_sha(
                locks.get("evaluation_suite_sha256"), "evaluation_suite_sha256"
            ),
        ),
        identity=_Identity(
            model_route=ModelRoute(_text(identity.get("model_route"), "model_route")),
            run_profile=RunProfile(_text(identity.get("run_profile"), "run_profile")),
            stage=Stage(_text(identity.get("stage"), "stage")),
            parent_run_id=_text(identity.get("parent_run_id"), "parent_run_id"),
            parent_checkpoint_step=_positive_int(
                identity.get("parent_checkpoint_step"), "parent_checkpoint_step"
            ),
            parent_weights_sha256=_sha(
                identity.get("parent_weights_sha256"), "parent_weights_sha256"
            ),
            parent_model_config_sha256=_sha(
                identity.get("parent_model_config_sha256"),
                "parent_model_config_sha256",
            ),
            tokenizer_sha256=_sha(identity.get("tokenizer_sha256"), "tokenizer_sha256"),
            data_manifest_sha256=_sha(
                identity.get("data_manifest_sha256"), "data_manifest_sha256"
            ),
        ),
        training=_Training(
            budget_mode=_text(training.get("budget_mode"), "budget_mode"),
            global_step=_positive_int(training.get("global_step"), "global_step"),
            target_train_tokens=_positive_int(
                training.get("target_train_tokens"), "target_train_tokens"
            ),
            expected_tokens_seen=_positive_int(
                training.get("expected_tokens_seen"), "expected_tokens_seen"
            ),
            minimum_target_token_coverage=_number(
                training.get("minimum_target_token_coverage"),
                "minimum_target_token_coverage",
            ),
            maximum_target_token_coverage=_number(
                training.get("maximum_target_token_coverage"),
                "maximum_target_token_coverage",
            ),
            required_training_metric_keys=_string_tuple(
                training.get("required_training_metric_keys"),
                "required_training_metric_keys",
            ),
        ),
        evaluation=_Evaluation(
            dev_suite=_text(evaluation.get("dev_suite"), "dev_suite"),
            dev_examples=_positive_int(evaluation.get("dev_examples"), "dev_examples"),
            test_suite=_text(evaluation.get("test_suite"), "test_suite"),
            test_examples=_positive_int(
                evaluation.get("test_examples"), "test_examples"
            ),
            required_metric_keys=_string_tuple(
                evaluation.get("required_metric_keys"), "required_metric_keys"
            ),
        ),
        runtime=_Runtime(
            platform_system=_text(runtime.get("platform_system"), "platform_system"),
            accelerator_type=_text(runtime.get("accelerator_type"), "accelerator_type"),
            require_clean_source=_boolean(
                runtime.get("require_clean_source"), "require_clean_source"
            ),
        ),
    )
    if (
        spec.training.budget_mode != "max_train_tokens"
        or spec.training.minimum_target_token_coverage <= 0
        or spec.training.maximum_target_token_coverage
        < spec.training.minimum_target_token_coverage
    ):
        raise ConfigError("invalid SFT run training acceptance bounds")
    return spec


def _check(
    name: str,
    passed: bool,
    passed_message: str,
    failed_message: str,
    details: Mapping[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(
        name=name,
        status=CheckStatus.PASS if passed else CheckStatus.FAIL,
        message=passed_message if passed else failed_message,
        details=details or {},
    )


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        raise ArtifactError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must be a JSON object: {path}")
    return value


def _load_contract(
    path: Path,
    loader: Callable[[Mapping[str, Any]], Any],
    name: str,
) -> Any:
    try:
        return loader(_load_json(path, name))
    except (KeyError, TypeError, ValueError, ContractError) as exc:
        raise ArtifactError(f"invalid {name} {path}: {exc}") from exc


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ArtifactError(f"metrics line {line_number} must be a JSON object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read metrics {path}: {exc}") from exc
    return rows


def _language_loss_is_consistent(value: Mapping[str, Any]) -> bool:
    loss = _as_float(value.get("eval_loss"))
    en_loss = _as_float(value.get("eval_en_loss"))
    zh_loss = _as_float(value.get("eval_zh_loss"))
    tokens = _as_float(value.get("eval_tokens"))
    en_tokens = _as_float(value.get("eval_en_tokens"))
    zh_tokens = _as_float(value.get("eval_zh_tokens"))
    if (
        None in {loss, en_loss, zh_loss, tokens, en_tokens, zh_tokens}
        or tokens <= 0
        or en_tokens <= 0
        or zh_tokens <= 0
        or not math.isclose(tokens, en_tokens + zh_tokens, abs_tol=1e-9)
    ):
        return False
    weighted = (en_loss * en_tokens + zh_loss * zh_tokens) / tokens
    return math.isclose(loss, weighted, rel_tol=0.0, abs_tol=1e-6)


def _safe_child(base: Path, value: Any, run: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else run / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(base.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def _config_path(value: Mapping[str, Any], name: str, root: Path) -> Path:
    return _resolve_path(_text(value.get(name), f"data.{name}"), root)


def _spec_path(value: Mapping[str, Any], name: str, root: Path) -> Path:
    return _resolve_path(_text(value.get(name), name), root)


def _resolve_path(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _strict(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"SFT run gate {name} has unknown {', '.join(unknown)}")
    missing = sorted(allowed - set(value))
    if missing:
        raise ConfigError(f"SFT run gate {name} is missing {', '.join(missing)}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"SFT run gate {name} must be a mapping")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"SFT run gate {name} must be non-empty text")
    return value


def _sha(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ConfigError(f"SFT run gate {name} must be a SHA-256")
    return text


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"SFT run gate {name} must be a positive integer")
    return value


def _number(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ConfigError(f"SFT run gate {name} must be finite")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"SFT run gate {name} must be boolean")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ConfigError(f"SFT run gate {name} must be unique non-empty strings")
    return tuple(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite(value: Any) -> bool:
    return _as_float(value) is not None


def _positive_finite(value: Any) -> bool:
    result = _as_float(value)
    return result is not None and result > 0


def _has_failures(checks: list[CheckResult]) -> bool:
    return any(check.status is CheckStatus.FAIL for check in checks)
