"""Verify the frozen public-data boundary for Native-60M SFT."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_lifecycle_lab.config import load_mapping
from llm_lifecycle_lab.contracts import (
    SCHEMA_VERSION,
    DataManifest,
    JsonContract,
    RecordKind,
    utc_now,
)
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.prepare import (
    load_data_manifest,
    verify_data_manifest,
)
from llm_lifecycle_lab.data.schemas import (
    format_validation_failure,
    validate_jsonl,
)
from llm_lifecycle_lab.data.split import SplitRatios, group_value, split_records
from llm_lifecycle_lab.doctor.result import CheckResult, CheckStatus
from llm_lifecycle_lab.exceptions import (
    ArtifactError,
    ConfigError,
    ContractError,
    DataValidationError,
    LLMLabError,
)

if TYPE_CHECKING:
    from llm_lifecycle_lab.tokenizer import NativeTokenizer

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_FORMATS = frozenset({"json", "jsonl", "parquet"})
_REQUIRED_SPLITS = ("train", "dev", "test")
_TASK_FAMILIES = (
    "general",
    "short_qa",
    "classification",
    "numeric",
    "structured",
)
_SHORT_QA_TASKS = frozenset(
    {"closed_qa", "information_extraction", "nlpcc_dbqa", "short_qa"}
)
_SELECTION_KEYS = {
    "oasst1": {
        "strategy",
        "records_per_language",
        "max_messages",
        "max_characters",
        "require_all_assistant_rank_zero",
        "reject_synthetic",
    },
    "dolly": {
        "strategy",
        "general_records",
        "classification_records",
        "short_qa_records",
        "classification_max_answer_characters",
        "short_qa_max_answer_characters",
        "max_characters",
    },
    "hc3-chinese": {"strategy", "records", "max_characters"},
    "squad": {
        "strategy",
        "records",
        "json_integer_records",
        "json_string_records",
        "max_context_characters",
        "max_answer_characters",
        "max_characters",
    },
    "cmrc2018": {
        "strategy",
        "records",
        "json_integer_records",
        "json_string_records",
        "max_context_characters",
        "max_answer_characters",
        "max_characters",
    },
    "msvamp": {
        "strategy",
        "expected_groups",
        "sft_warmup_groups",
        "reserved_groups",
    },
}
_SOURCE_TRANSFORMS = {
    "oasst1": "oasst-ranked-conversation-v1",
    "dolly": "dolly-balanced-sft-v1",
    "hc3-chinese": "hc3-human-single-turn-sft-v1",
    "squad": "squad-extractive-sft-v1",
    "cmrc2018": "cmrc-extractive-sft-v1",
    "msvamp": "msvamp-parallel-integer-v1",
}


@dataclass(frozen=True, slots=True)
class SFTDataGateReport(JsonContract):
    gate_id: str
    root: str
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
class _RecipeSource:
    source_id: str
    transform: str
    provider: str
    repository: str
    revision: str
    file: str
    file_sha256: str
    file_format: str
    license: str
    split: str
    selection: Mapping[str, Any]

    @property
    def uri(self) -> str:
        return f"hf://datasets/{self.repository}@{self.revision}/{self.file}"


@dataclass(frozen=True, slots=True)
class _Ratios:
    train: float
    dev: float
    test: float

    def as_split_ratios(self) -> SplitRatios:
        return SplitRatios(train=self.train, dev=self.dev, test=self.test)


@dataclass(frozen=True, slots=True)
class _RecipePreparation:
    dataset_id: str
    split_seed: int
    group_by: str
    ratios: _Ratios
    license: str


@dataclass(frozen=True, slots=True)
class _SFTDataRecipe:
    recipe_id: str
    selection_namespace: str
    description: str
    sources: tuple[_RecipeSource, ...]
    output_record_kind: str
    output_records: int
    output_sha256: str
    preparation: _RecipePreparation
    reserved: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class _Locks:
    base_gate: Path
    base_gate_sha256: str
    recipe: Path
    recipe_sha256: str


@dataclass(frozen=True, slots=True)
class _Paths:
    canonical_source: Path
    prepared_manifest: Path
    tokenizer: Path


@dataclass(frozen=True, slots=True)
class _Identity:
    recipe_id: str
    selection_namespace: str
    dataset_id: str
    record_kind: str
    source_records: int
    source_sha256: str
    tokenizer_id: str
    tokenizer_sha256: str
    tokenizer_manifest_sha256: str
    chat_template_version: str


@dataclass(frozen=True, slots=True)
class _Preparation:
    split_seed: int
    group_by: str
    ratios: _Ratios
    license: str


@dataclass(frozen=True, slots=True)
class _Quality:
    required_languages: tuple[str, ...]
    max_sequence_length: int
    train_examples: int
    train_supervised_tokens: int
    train_language_examples: Mapping[str, int]
    train_task_examples: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _Policy:
    public_only: bool
    allow_synthetic: bool
    require_pinned_revisions: bool
    require_pinned_file_hashes: bool
    allowed_source_ids: tuple[str, ...]
    excluded_source_ids: tuple[str, ...]
    supervision: str
    truncation: str
    duplicate_normalization: str
    require_clean_git: bool


@dataclass(frozen=True, slots=True)
class _SFTDataSpec:
    gate_id: str
    locks: _Locks
    paths: _Paths
    identity: _Identity
    preparation: _Preparation
    quality: _Quality
    policy: _Policy
    recipe: _SFTDataRecipe


def verify_sft_data_gate(
    spec_path: str | Path,
    *,
    workdir: str | Path = ".",
    check_inputs: bool = False,
    check_git: bool = False,
) -> SFTDataGateReport:
    """Verify the SFT data contract, materialized inputs, or full preflight."""

    if check_git and not check_inputs:
        raise ValueError("check_git requires check_inputs")
    root = Path(workdir).resolve()
    scope = (
        "preflight"
        if check_git
        else "inputs-only"
        if check_inputs
        else ("contract-only")
    )
    spec = _load_spec(Path(spec_path), root=root)
    checks = [
        _file_lock_check(
            "base-gate-lock",
            spec.locks.base_gate,
            spec.locks.base_gate_sha256,
            "SFT Base initialization gate is byte-for-byte frozen",
        ),
        _file_lock_check(
            "recipe-lock",
            spec.locks.recipe,
            spec.locks.recipe_sha256,
            "SFT public-data recipe is byte-for-byte frozen",
        ),
        _recipe_identity_check(spec),
        _source_policy_check(spec),
        _processing_policy_check(spec),
    ]
    if _has_failures(checks) or not check_inputs:
        return _report(spec, root, scope, checks)

    checks.append(_base_initialization_check(spec, root))
    source_records, source_check = _load_canonical_source(spec)
    checks.append(source_check)
    manifest, manifest_check = _load_prepared_manifest(spec)
    checks.append(manifest_check)
    tokenizer, tokenizer_check = _load_tokenizer(spec)
    checks.append(tokenizer_check)
    if (
        _has_failures(checks)
        or source_records is None
        or manifest is None
        or tokenizer is None
    ):
        return _report(spec, root, scope, checks)

    checks.append(_split_artifact_check(spec))
    if not _has_failures(checks):
        checks.extend(
            _prepared_content_checks(
                spec,
                manifest,
                source_records,
                tokenizer,
            )
        )
    if check_git and not _has_failures(checks):
        checks.append(_clean_git_check(root, spec.policy.require_clean_git))
    return _report(spec, root, scope, checks)


def _report(
    spec: _SFTDataSpec,
    root: Path,
    scope: str,
    checks: list[CheckResult],
) -> SFTDataGateReport:
    return SFTDataGateReport(
        gate_id=spec.gate_id,
        root=str(root),
        checks=tuple(checks),
        scope=scope,
    )


def _file_lock_check(
    name: str,
    path: Path,
    expected_sha256: str,
    passed: str,
) -> CheckResult:
    actual = sha256_file(path) if path.is_file() and not path.is_symlink() else None
    return _check(
        name,
        actual == expected_sha256,
        passed,
        f"{name} does not match the frozen SHA-256",
        {
            "path": str(path),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual,
        },
    )


def _recipe_identity_check(spec: _SFTDataSpec) -> CheckResult:
    recipe = spec.recipe
    identity = spec.identity
    preparation = spec.preparation
    matches = (
        recipe.recipe_id == identity.recipe_id
        and recipe.selection_namespace == identity.selection_namespace
        and recipe.output_record_kind == identity.record_kind == RecordKind.SFT.value
        and recipe.output_records == identity.source_records
        and recipe.output_sha256 == identity.source_sha256
        and recipe.preparation.dataset_id == identity.dataset_id
        and recipe.preparation.split_seed == preparation.split_seed
        and recipe.preparation.group_by == preparation.group_by
        and recipe.preparation.ratios == preparation.ratios
        and recipe.preparation.license == preparation.license
    )
    return _check(
        "recipe-identity",
        matches,
        "recipe output and preparation identity match the data gate",
        "recipe output or preparation identity drifted from the data gate",
        {
            "recipe_id": recipe.recipe_id,
            "selection_namespace": recipe.selection_namespace,
            "dataset_id": recipe.preparation.dataset_id,
            "source_records": recipe.output_records,
            "source_sha256": recipe.output_sha256,
        },
    )


def _source_policy_check(spec: _SFTDataSpec) -> CheckResult:
    recipe = spec.recipe
    policy = spec.policy
    source_ids = tuple(source.source_id for source in recipe.sources)
    excluded = set(policy.excluded_source_ids)
    issues: list[str] = []
    if not policy.public_only:
        issues.append("public_only must remain enabled")
    if policy.allow_synthetic:
        issues.append("synthetic SFT data must remain disabled")
    if not policy.require_pinned_revisions:
        issues.append("source revisions must remain pinned")
    if not policy.require_pinned_file_hashes:
        issues.append("upstream file hashes must remain pinned")
    if source_ids != policy.allowed_source_ids:
        issues.append("recipe sources do not exactly match the ordered allowlist")
    if source_ids != tuple(_SOURCE_TRANSFORMS):
        issues.append("recipe sources do not match the approved SFT source set")
    if set(source_ids) & excluded:
        issues.append("an excluded source is present in the SFT recipe")
    if len(source_ids) != len(set(source_ids)):
        issues.append("recipe contains duplicate source IDs")
    for source in recipe.sources:
        if _SOURCE_TRANSFORMS.get(source.source_id) != source.transform:
            issues.append(f"{source.source_id}: transform is not approved")
        if policy.public_only and source.provider != "huggingface":
            issues.append(f"{source.source_id}: provider is not huggingface")
        if policy.require_pinned_revisions and not _COMMIT_PATTERN.fullmatch(
            source.revision
        ):
            issues.append(f"{source.source_id}: revision is not a commit")
        if policy.require_pinned_file_hashes and not _SHA256_PATTERN.fullmatch(
            source.file_sha256
        ):
            issues.append(f"{source.source_id}: upstream file hash is not pinned")
        if source.file_format not in _SOURCE_FORMATS:
            issues.append(f"{source.source_id}: unsupported source format")
    source_licenses = {source.license for source in recipe.sources}
    declared_licenses = set(spec.preparation.license.split(" AND "))
    if source_licenses != declared_licenses:
        issues.append("composite license does not match source licenses")
    oasst = next(
        (source for source in recipe.sources if source.source_id == "oasst1"),
        None,
    )
    if not policy.allow_synthetic and (
        oasst is None or oasst.selection.get("reject_synthetic") is not True
    ):
        issues.append("OASST1 synthetic rows are not explicitly rejected")
    if (
        oasst is None
        or oasst.selection.get("require_all_assistant_rank_zero") is not True
    ):
        issues.append("OASST1 assistant rank-zero policy is not enforced")
    helpsteer = recipe.reserved.get("helpsteer3")
    if (
        "helpsteer3" not in excluded
        or helpsteer is None
        or helpsteer.get("stage") != "dpo"
        or set(recipe.reserved) != {"helpsteer3"}
        or policy.excluded_source_ids != ("helpsteer3",)
    ):
        issues.append("HelpSteer3 must remain reserved for DPO")
    return _check(
        "public-source-policy",
        not issues,
        "only pinned public SFT sources are allowed; preference data is excluded",
        "SFT source policy is not satisfied",
        {"source_ids": list(source_ids), "issues": issues},
    )


def _processing_policy_check(spec: _SFTDataSpec) -> CheckResult:
    policy = spec.policy
    quality = spec.quality
    ratios = spec.preparation.ratios
    matches = (
        spec.preparation.group_by == "source_id"
        and spec.preparation.split_seed == 42
        and math.isclose(ratios.train, 0.8)
        and math.isclose(ratios.dev, 0.1)
        and math.isclose(ratios.test, 0.1)
        and quality.required_languages == ("en", "zh")
        and quality.max_sequence_length == 512
        and policy.supervision == "assistant-only"
        and policy.truncation == "forbidden"
        and policy.duplicate_normalization == "NFKC-whitespace-collapse-v1"
        and policy.require_clean_git is True
    )
    return _check(
        "data-processing-policy",
        matches,
        "group split, bilingual coverage, supervision and truncation are frozen",
        "SFT data processing policy was weakened or changed",
        {
            "split_seed": spec.preparation.split_seed,
            "group_by": spec.preparation.group_by,
            "ratios": {
                "train": ratios.train,
                "dev": ratios.dev,
                "test": ratios.test,
            },
            "supervision": policy.supervision,
            "truncation": policy.truncation,
        },
    )


def _base_initialization_check(spec: _SFTDataSpec, root: Path) -> CheckResult:
    try:
        from llm_lifecycle_lab.sft_initialization import verify_sft_initialization

        report = verify_sft_initialization(
            spec.locks.base_gate,
            workdir=root,
            check_git=False,
        )
        matches = not report.has_failures
        details: Mapping[str, Any] = report.to_dict()
    except (ImportError, LLMLabError, OSError, ValueError) as exc:
        matches = False
        details = {"error": str(exc)}
    return _check(
        "base-initialization",
        matches,
        "Stage 0 canonical Base initialization gate passes",
        "Stage 0 canonical Base initialization gate failed",
        details,
    )


def _load_canonical_source(
    spec: _SFTDataSpec,
) -> tuple[tuple[dict[str, Any], ...] | None, CheckResult]:
    path = spec.paths.canonical_source
    details: dict[str, Any] = {"path": str(path)}
    records: tuple[dict[str, Any], ...] | None = None
    try:
        if not path.is_file() or path.is_symlink():
            raise DataValidationError(f"canonical source does not exist: {path}")
        details["actual_sha256"] = sha256_file(path)
        validated = validate_jsonl(path, RecordKind.SFT)
        details["records"] = validated.report.records_seen
        if not validated.report.ok:
            raise DataValidationError(format_validation_failure(validated.report))
        records = validated.records
        matches = (
            details["actual_sha256"] == spec.identity.source_sha256
            and len(records) == spec.identity.source_records
        )
    except (OSError, LLMLabError) as exc:
        matches = False
        details["error"] = str(exc)
    return records, _check(
        "canonical-source",
        matches,
        "canonical SFT source hash, count and schema match",
        "canonical SFT source is missing, malformed or drifted",
        details,
    )


def _load_prepared_manifest(
    spec: _SFTDataSpec,
) -> tuple[DataManifest | None, CheckResult]:
    path = spec.paths.prepared_manifest
    details: dict[str, Any] = {"path": str(path)}
    manifest: DataManifest | None = None
    try:
        manifest = load_data_manifest(path)
        source_size = (
            spec.paths.canonical_source.stat().st_size
            if spec.paths.canonical_source.is_file()
            else None
        )
        processing_steps = (
            "validate-schema",
            "deterministic-group-split:0.8:0.1:0.1",
        )
        matches = (
            manifest.dataset_id == spec.identity.dataset_id
            and manifest.record_kind is RecordKind.SFT
            and manifest.source.sha256 == spec.identity.source_sha256
            and manifest.source.size_bytes == source_size
            and manifest.split_seed == spec.preparation.split_seed
            and manifest.group_by == spec.preparation.group_by
            and manifest.license == spec.preparation.license
            and manifest.processing_steps == processing_steps
            and tuple(split.name for split in manifest.splits) == _REQUIRED_SPLITS
            and sum(split.records for split in manifest.splits)
            == spec.identity.source_records
        )
        details.update(
            {
                "dataset_id": manifest.dataset_id,
                "record_kind": manifest.record_kind.value,
                "source_sha256": manifest.source.sha256,
                "split_seed": manifest.split_seed,
                "group_by": manifest.group_by,
                "license": manifest.license,
                "split_records": {
                    split.name: split.records for split in manifest.splits
                },
            }
        )
    except (OSError, LLMLabError) as exc:
        matches = False
        details["error"] = str(exc)
    return manifest, _check(
        "prepared-manifest",
        matches,
        "prepared SFT manifest matches source, split and license identity",
        "prepared SFT manifest is missing, malformed or drifted",
        details,
    )


def _load_tokenizer(
    spec: _SFTDataSpec,
) -> tuple[NativeTokenizer | None, CheckResult]:
    root = spec.paths.tokenizer
    tokenizer_path = root / "tokenizer.json"
    manifest_path = root / "tokenizer_manifest.json"
    details: dict[str, Any] = {"path": str(root)}
    tokenizer: NativeTokenizer | None = None
    try:
        from llm_lifecycle_lab.tokenizer import NativeTokenizer

        tokenizer_hash = sha256_file(tokenizer_path)
        manifest_hash = sha256_file(manifest_path)
        tokenizer = NativeTokenizer.from_directory(root)
        matches = (
            tokenizer_hash == spec.identity.tokenizer_sha256
            and manifest_hash == spec.identity.tokenizer_manifest_sha256
            and tokenizer.manifest.tokenizer_id == spec.identity.tokenizer_id
            and tokenizer.chat_template_version == spec.identity.chat_template_version
        )
        details.update(
            {
                "tokenizer_sha256": tokenizer_hash,
                "tokenizer_manifest_sha256": manifest_hash,
                "tokenizer_id": tokenizer.manifest.tokenizer_id,
                "chat_template_version": tokenizer.chat_template_version,
            }
        )
    except (ImportError, OSError, ArtifactError, ContractError) as exc:
        matches = False
        details["error"] = str(exc)
    return tokenizer, _check(
        "tokenizer-identity",
        matches,
        "canonical Base tokenizer matches the SFT data contract",
        "SFT tokenizer is missing, incompatible or drifted",
        details,
    )


def _split_artifact_check(
    spec: _SFTDataSpec,
) -> CheckResult:
    try:
        failures = verify_data_manifest(spec.paths.prepared_manifest)
    except (OSError, DataValidationError) as exc:
        failures = [str(exc)]
    return _check(
        "split-artifacts",
        not failures,
        "prepared split hashes and record counts match the manifest",
        "prepared split artifacts are missing or drifted",
        {"failures": failures},
    )


def _prepared_content_checks(
    spec: _SFTDataSpec,
    manifest: DataManifest,
    source_records: tuple[dict[str, Any], ...],
    tokenizer: NativeTokenizer,
) -> list[CheckResult]:
    records_by_split: dict[str, tuple[dict[str, Any], ...]] = {}
    schema_errors: list[str] = []
    for split in manifest.splits:
        path = spec.paths.prepared_manifest.parent / split.path
        validated = validate_jsonl(path, RecordKind.SFT)
        if not validated.report.ok:
            schema_errors.append(format_validation_failure(validated.report))
        records_by_split[split.name] = validated.records
    schema_check = _check(
        "split-schema",
        not schema_errors,
        "all prepared splits satisfy the SFT JSONL schema",
        "one or more prepared splits violate the SFT JSONL schema",
        {"errors": schema_errors[:10]},
    )
    if schema_errors:
        return [schema_check]

    expected = split_records(
        source_records,
        ratios=spec.preparation.ratios.as_split_ratios(),
        seed=spec.preparation.split_seed,
        group_by=spec.preparation.group_by,
    )
    assignment_errors: list[str] = []
    split_manifests = {split.name: split for split in manifest.splits}
    for split_name in _REQUIRED_SPLITS:
        actual = records_by_split[split_name]
        declared = split_manifests[split_name]
        groups = {group_value(record, spec.preparation.group_by) for record in actual}
        if list(actual) != expected[split_name]:
            assignment_errors.append(
                f"{split_name}: records differ from deterministic source split"
            )
        if len(actual) != declared.records:
            assignment_errors.append(
                f"{split_name}: loaded records differ from manifest"
            )
        if len(groups) != declared.groups:
            assignment_errors.append(f"{split_name}: group count differs from manifest")
    assignment_check = _check(
        "deterministic-split",
        not assignment_errors,
        "prepared rows exactly match the frozen source_id group split",
        "prepared rows do not match the deterministic source split",
        {"errors": assignment_errors[:10]},
    )

    protocol_errors: list[str] = []
    conversation_owners: dict[str, str] = {}
    language_counts: dict[str, Counter[str]] = {}
    task_counts: dict[str, Counter[str]] = {}
    supervised_counts: dict[str, int] = {}
    recipe_sources = {source.source_id: source for source in spec.recipe.sources}
    for split_name in _REQUIRED_SPLITS:
        language_counts[split_name] = Counter()
        task_counts[split_name] = Counter()
        supervised_counts[split_name] = 0
        for record in records_by_split[split_name]:
            error_prefix = f"{split_name}:{record.get('id', '<unknown>')}"
            try:
                language = _required_record_language(
                    record,
                    spec.quality.required_languages,
                )
                source = _record_source(record, recipe_sources)
                messages = _record_messages(record)
                supervised = _supervised_token_count(
                    messages,
                    tokenizer=tokenizer,
                    max_sequence_length=spec.quality.max_sequence_length,
                )
                normalized = _normalized_conversation(messages)
                if normalized in conversation_owners:
                    raise ContractError("duplicate normalized SFT conversation")
                conversation_owners[normalized] = split_name
                if record.get("synthetic") not in (None, False):
                    raise ContractError("synthetic SFT record is forbidden")
                metadata = record.get("metadata")
                if isinstance(metadata, Mapping) and metadata.get("synthetic") not in (
                    None,
                    False,
                ):
                    raise ContractError("synthetic SFT record is forbidden")
                language_counts[split_name][language] += 1
                task_counts[split_name][_task_family(record)] += 1
                supervised_counts[split_name] += supervised
                if source.source_id in spec.policy.excluded_source_ids:
                    raise ContractError("excluded source entered SFT data")
            except (ContractError, DataValidationError) as exc:
                if len(protocol_errors) < 20:
                    protocol_errors.append(f"{error_prefix}: {exc}")

    required_languages = set(spec.quality.required_languages)
    for split_name in _REQUIRED_SPLITS:
        if set(language_counts[split_name]) != required_languages:
            protocol_errors.append(f"{split_name}: must contain exactly en and zh")
    protocol_check = _check(
        "conversation-protocol",
        not protocol_errors,
        "all rows satisfy provenance, dialogue, language and length policies",
        "one or more SFT rows violate provenance or conversation policy",
        {"errors": protocol_errors[:20]},
    )

    train_languages = dict(language_counts["train"])
    train_tasks = dict(task_counts["train"])
    train_examples = len(records_by_split["train"])
    balance_matches = (
        train_examples == spec.quality.train_examples
        and supervised_counts["train"] == spec.quality.train_supervised_tokens
        and train_languages == dict(spec.quality.train_language_examples)
        and train_tasks == dict(spec.quality.train_task_examples)
    )
    balance_check = _check(
        "train-balance",
        balance_matches,
        "train examples, supervised tokens, languages and tasks match the freeze",
        "prepared train balance differs from the frozen SFT contract",
        {
            "examples": train_examples,
            "supervised_tokens": supervised_counts["train"],
            "language_examples": train_languages,
            "task_examples": train_tasks,
        },
    )
    return [schema_check, assignment_check, protocol_check, balance_check]


def _required_record_language(
    record: Mapping[str, Any],
    allowed: Sequence[str],
) -> str:
    language = record.get("language")
    if not isinstance(language, str) or language not in allowed:
        raise ContractError("SFT record language must be en or zh")
    return language


def _record_source(
    record: Mapping[str, Any],
    sources: Mapping[str, _RecipeSource],
) -> _RecipeSource:
    source_id = record.get("source_id")
    if not isinstance(source_id, str) or ":" not in source_id:
        raise ContractError("SFT source_id must contain a declared source prefix")
    prefix, suffix = source_id.split(":", maxsplit=1)
    if not suffix or prefix not in sources:
        raise ContractError("SFT source_id prefix is not declared by the recipe")
    source = sources[prefix]
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ContractError("SFT metadata must be an object")
    if metadata.get("transform") != source.transform:
        raise ContractError("SFT metadata transform does not match its source")
    if metadata.get("source") != source.uri:
        raise ContractError("SFT metadata source URI does not match its recipe")
    return source


def _record_messages(
    record: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ContractError("SFT messages must be a non-empty list")
    if not all(isinstance(message, Mapping) for message in messages):
        raise ContractError("SFT messages must contain objects")
    return list(messages)


def _supervised_token_count(
    messages: list[Mapping[str, Any]],
    *,
    tokenizer: NativeTokenizer,
    max_sequence_length: int,
) -> int:
    if messages[-1].get("role") != "assistant":
        raise ContractError("SFT conversation must end with assistant")
    token_ids = tokenizer.encode_chat(messages)
    if len(token_ids) > max_sequence_length:
        raise ContractError(
            f"SFT conversation has {len(token_ids)} tokens; truncation is forbidden"
        )
    starts = [
        index
        for index, token_id in enumerate(token_ids)
        if token_id == tokenizer.chat_start_token_id
    ]
    ends = [
        index
        for index, token_id in enumerate(token_ids)
        if token_id == tokenizer.chat_end_token_id
    ]
    if len(starts) != len(messages) or len(ends) != len(messages):
        raise ContractError("chat control token boundaries do not match messages")
    supervised = 0
    for message, start, end in zip(messages, starts, ends, strict=True):
        role = message.get("role")
        if not isinstance(role, str):
            raise ContractError("SFT role must be text")
        header = tokenizer.encode(role + "\n")
        body = start + 1 + len(header)
        if token_ids[start + 1 : body] != header or body >= end:
            raise ContractError("SFT header/body token boundary is ambiguous")
        if role == "assistant":
            supervised += end - body + 1
    if supervised <= 0:
        raise ContractError("SFT conversation has no supervised assistant tokens")
    return supervised


def _normalized_conversation(messages: list[Mapping[str, Any]]) -> str:
    value = [
        {
            "role": message.get("role"),
            "content": _normalize(str(message.get("content", ""))),
        }
        for message in messages
    ]
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _task_family(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    values = metadata if isinstance(metadata, Mapping) else {}
    task = str(values.get("task", "")).strip()
    transform = str(values.get("transform", "")).strip()
    if task == "classification":
        return "classification"
    if task in _SHORT_QA_TASKS:
        return "short_qa"
    if task == "structured":
        return "structured"
    if task in {"math-verifiable", "numeric"} or (
        transform == "msvamp-parallel-integer-v1"
    ):
        return "numeric"
    return "general"


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


def _load_spec(path: Path, *, root: Path) -> _SFTDataSpec:
    value = load_mapping(path)
    _exact_keys(
        value,
        {
            "schema_version",
            "gate_id",
            "locks",
            "paths",
            "identity",
            "preparation",
            "quality",
            "policy",
        },
        "SFT data gate",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ConfigError("SFT data gate has unsupported schema_version")
    locks = _load_locks(value["locks"], root)
    return _SFTDataSpec(
        gate_id=_text(value["gate_id"], "gate_id"),
        locks=locks,
        paths=_load_paths(value["paths"], root),
        identity=_load_identity(value["identity"]),
        preparation=_load_preparation(value["preparation"]),
        quality=_load_quality(value["quality"]),
        policy=_load_policy(value["policy"]),
        recipe=_load_recipe(locks.recipe),
    )


def _load_locks(value: Any, root: Path) -> _Locks:
    mapping = _required_mapping(value, "locks")
    _exact_keys(
        mapping,
        {
            "base_gate",
            "base_gate_sha256",
            "recipe",
            "recipe_sha256",
        },
        "locks",
    )
    return _Locks(
        base_gate=_root_path(root, _text(mapping["base_gate"], "locks.base_gate")),
        base_gate_sha256=_sha256(mapping["base_gate_sha256"], "locks.base_gate_sha256"),
        recipe=_root_path(root, _text(mapping["recipe"], "locks.recipe")),
        recipe_sha256=_sha256(mapping["recipe_sha256"], "locks.recipe_sha256"),
    )


def _load_paths(value: Any, root: Path) -> _Paths:
    mapping = _required_mapping(value, "paths")
    keys = {"canonical_source", "prepared_manifest", "tokenizer"}
    _exact_keys(mapping, keys, "paths")
    return _Paths(
        **{key: _root_path(root, _text(mapping[key], f"paths.{key}")) for key in keys}
    )


def _load_identity(value: Any) -> _Identity:
    mapping = _required_mapping(value, "identity")
    keys = {
        "recipe_id",
        "selection_namespace",
        "dataset_id",
        "record_kind",
        "source_records",
        "source_sha256",
        "tokenizer_id",
        "tokenizer_sha256",
        "tokenizer_manifest_sha256",
        "chat_template_version",
    }
    _exact_keys(mapping, keys, "identity")
    return _Identity(
        recipe_id=_text(mapping["recipe_id"], "identity.recipe_id"),
        selection_namespace=_text(
            mapping["selection_namespace"], "identity.selection_namespace"
        ),
        dataset_id=_text(mapping["dataset_id"], "identity.dataset_id"),
        record_kind=_text(mapping["record_kind"], "identity.record_kind"),
        source_records=_integer(
            mapping["source_records"], "identity.source_records", minimum=1
        ),
        source_sha256=_sha256(mapping["source_sha256"], "identity.source_sha256"),
        tokenizer_id=_text(mapping["tokenizer_id"], "identity.tokenizer_id"),
        tokenizer_sha256=_sha256(
            mapping["tokenizer_sha256"], "identity.tokenizer_sha256"
        ),
        tokenizer_manifest_sha256=_sha256(
            mapping["tokenizer_manifest_sha256"],
            "identity.tokenizer_manifest_sha256",
        ),
        chat_template_version=_text(
            mapping["chat_template_version"],
            "identity.chat_template_version",
        ),
    )


def _load_preparation(value: Any) -> _Preparation:
    mapping = _required_mapping(value, "preparation")
    keys = {"split_seed", "group_by", "ratios", "license"}
    _exact_keys(mapping, keys, "preparation")
    return _Preparation(
        split_seed=_integer(mapping["split_seed"], "preparation.split_seed", minimum=0),
        group_by=_text(mapping["group_by"], "preparation.group_by"),
        ratios=_load_ratios(mapping["ratios"], "preparation.ratios"),
        license=_text(mapping["license"], "preparation.license"),
    )


def _load_quality(value: Any) -> _Quality:
    mapping = _required_mapping(value, "quality")
    keys = {
        "required_languages",
        "max_sequence_length",
        "train_examples",
        "train_supervised_tokens",
        "train_language_examples",
        "train_task_examples",
    }
    _exact_keys(mapping, keys, "quality")
    languages = _text_sequence(
        mapping["required_languages"], "quality.required_languages"
    )
    language_examples = _count_mapping(
        mapping["train_language_examples"],
        "quality.train_language_examples",
        expected=set(languages),
    )
    task_examples = _count_mapping(
        mapping["train_task_examples"],
        "quality.train_task_examples",
        expected=set(_TASK_FAMILIES),
    )
    train_examples = _integer(
        mapping["train_examples"], "quality.train_examples", minimum=1
    )
    if sum(language_examples.values()) != train_examples:
        raise ConfigError("quality train language counts do not sum to examples")
    if sum(task_examples.values()) != train_examples:
        raise ConfigError("quality train task counts do not sum to examples")
    return _Quality(
        required_languages=languages,
        max_sequence_length=_integer(
            mapping["max_sequence_length"],
            "quality.max_sequence_length",
            minimum=2,
        ),
        train_examples=train_examples,
        train_supervised_tokens=_integer(
            mapping["train_supervised_tokens"],
            "quality.train_supervised_tokens",
            minimum=1,
        ),
        train_language_examples=language_examples,
        train_task_examples=task_examples,
    )


def _load_policy(value: Any) -> _Policy:
    mapping = _required_mapping(value, "policy")
    keys = {
        "public_only",
        "allow_synthetic",
        "require_pinned_revisions",
        "require_pinned_file_hashes",
        "allowed_source_ids",
        "excluded_source_ids",
        "supervision",
        "truncation",
        "duplicate_normalization",
        "require_clean_git",
    }
    _exact_keys(mapping, keys, "policy")
    return _Policy(
        public_only=_boolean(mapping["public_only"], "policy.public_only"),
        allow_synthetic=_boolean(mapping["allow_synthetic"], "policy.allow_synthetic"),
        require_pinned_revisions=_boolean(
            mapping["require_pinned_revisions"],
            "policy.require_pinned_revisions",
        ),
        require_pinned_file_hashes=_boolean(
            mapping["require_pinned_file_hashes"],
            "policy.require_pinned_file_hashes",
        ),
        allowed_source_ids=_text_sequence(
            mapping["allowed_source_ids"], "policy.allowed_source_ids"
        ),
        excluded_source_ids=_text_sequence(
            mapping["excluded_source_ids"], "policy.excluded_source_ids"
        ),
        supervision=_text(mapping["supervision"], "policy.supervision"),
        truncation=_text(mapping["truncation"], "policy.truncation"),
        duplicate_normalization=_text(
            mapping["duplicate_normalization"],
            "policy.duplicate_normalization",
        ),
        require_clean_git=_boolean(
            mapping["require_clean_git"], "policy.require_clean_git"
        ),
    )


def _load_recipe(path: Path) -> _SFTDataRecipe:
    value = load_mapping(path)
    _exact_keys(
        value,
        {
            "schema_version",
            "recipe_id",
            "selection_namespace",
            "description",
            "sources",
            "output",
            "preparation",
            "reserved",
        },
        "SFT data recipe",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ConfigError("SFT data recipe has unsupported schema_version")
    raw_sources = value["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ConfigError("SFT data recipe sources must be a non-empty list")
    sources = tuple(
        _load_recipe_source(item, index) for index, item in enumerate(raw_sources)
    )
    output = _required_mapping(value["output"], "recipe.output")
    _exact_keys(output, {"record_kind", "records", "sha256"}, "recipe.output")
    preparation = _required_mapping(value["preparation"], "recipe.preparation")
    _exact_keys(
        preparation,
        {"dataset_id", "split_seed", "group_by", "ratios", "license"},
        "recipe.preparation",
    )
    reserved_value = _required_mapping(value["reserved"], "recipe.reserved")
    reserved: dict[str, Mapping[str, str]] = {}
    for source_id, item_value in reserved_value.items():
        item = _required_mapping(item_value, f"recipe.reserved.{source_id}")
        _exact_keys(
            item,
            {"stage", "reason"},
            f"recipe.reserved.{source_id}",
        )
        reserved[str(source_id)] = {
            "stage": _text(item["stage"], f"recipe.reserved.{source_id}.stage"),
            "reason": _text(item["reason"], f"recipe.reserved.{source_id}.reason"),
        }
    return _SFTDataRecipe(
        recipe_id=_text(value["recipe_id"], "recipe.recipe_id"),
        selection_namespace=_text(
            value["selection_namespace"], "recipe.selection_namespace"
        ),
        description=_text(value["description"], "recipe.description"),
        sources=sources,
        output_record_kind=_text(output["record_kind"], "recipe.output.record_kind"),
        output_records=_integer(output["records"], "recipe.output.records", minimum=1),
        output_sha256=_sha256(output["sha256"], "recipe.output.sha256"),
        preparation=_RecipePreparation(
            dataset_id=_text(
                preparation["dataset_id"], "recipe.preparation.dataset_id"
            ),
            split_seed=_integer(
                preparation["split_seed"],
                "recipe.preparation.split_seed",
                minimum=0,
            ),
            group_by=_text(preparation["group_by"], "recipe.preparation.group_by"),
            ratios=_load_ratios(preparation["ratios"], "recipe.preparation.ratios"),
            license=_text(preparation["license"], "recipe.preparation.license"),
        ),
        reserved=reserved,
    )


def _load_recipe_source(value: Any, index: int) -> _RecipeSource:
    name = f"recipe.sources[{index}]"
    mapping = _required_mapping(value, name)
    keys = {
        "source_id",
        "transform",
        "provider",
        "repository",
        "revision",
        "file",
        "file_sha256",
        "format",
        "license",
        "split",
        "selection",
    }
    _exact_keys(mapping, keys, name)
    source_id = _text(mapping["source_id"], f"{name}.source_id")
    selection = _required_mapping(mapping["selection"], f"{name}.selection")
    expected_selection = _SELECTION_KEYS.get(source_id)
    if expected_selection is not None:
        _exact_keys(
            selection,
            expected_selection,
            f"{name}.selection",
        )
    if not selection:
        raise ConfigError(f"{name}.selection must not be empty")
    _validate_selection(source_id, selection, f"{name}.selection")
    return _RecipeSource(
        source_id=source_id,
        transform=_text(mapping["transform"], f"{name}.transform"),
        provider=_text(mapping["provider"], f"{name}.provider"),
        repository=_text(mapping["repository"], f"{name}.repository"),
        revision=_commit(mapping["revision"], f"{name}.revision"),
        file=_text(mapping["file"], f"{name}.file"),
        file_sha256=_sha256(mapping["file_sha256"], f"{name}.file_sha256"),
        file_format=_text(mapping["format"], f"{name}.format"),
        license=_text(mapping["license"], f"{name}.license"),
        split=_text(mapping["split"], f"{name}.split"),
        selection=dict(selection),
    )


def _validate_selection(
    source_id: str,
    selection: Mapping[str, Any],
    name: str,
) -> None:
    if source_id not in _SELECTION_KEYS:
        return
    expected_strategy = (
        "stable-hash-partition" if source_id == "msvamp" else "stable-hash-rank"
    )
    if selection.get("strategy") != expected_strategy:
        raise ConfigError(f"{name}.strategy must be {expected_strategy}")
    for key, value in selection.items():
        if key == "strategy":
            continue
        if key in {"require_all_assistant_rank_zero", "reject_synthetic"}:
            _boolean(value, f"{name}.{key}")
            continue
        minimum = 0 if key in {"json_integer_records", "json_string_records"} else 1
        _integer(value, f"{name}.{key}", minimum=minimum)
    if source_id == "msvamp":
        expected = int(selection["expected_groups"])
        selected = int(selection["sft_warmup_groups"])
        reserved = int(selection["reserved_groups"])
        if selected + reserved != expected:
            raise ConfigError(
                f"{name} warmup and reserved groups must sum to expected_groups"
            )


def _load_ratios(value: Any, name: str) -> _Ratios:
    mapping = _required_mapping(value, name)
    _exact_keys(mapping, {"train", "dev", "test"}, name)
    ratios = _Ratios(
        train=_nonnegative_number(mapping["train"], f"{name}.train"),
        dev=_nonnegative_number(mapping["dev"], f"{name}.dev"),
        test=_nonnegative_number(mapping["test"], f"{name}.test"),
    )
    if ratios.train <= 0 or not math.isclose(
        ratios.train + ratios.dev + ratios.test,
        1.0,
        abs_tol=1e-9,
    ):
        raise ConfigError(f"{name} must have positive train and sum to 1.0")
    return ratios


def _count_mapping(
    value: Any,
    name: str,
    *,
    expected: set[str],
) -> Mapping[str, int]:
    mapping = _required_mapping(value, name)
    _exact_keys(mapping, expected, name)
    return {
        str(key): _integer(item, f"{name}.{key}", minimum=1)
        for key, item in mapping.items()
    }


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ConfigError(f"{name} must be a non-empty list of strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ConfigError(f"{name} must not contain duplicates")
    return result


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
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ConfigError("frozen path must remain inside the repository")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConfigError("frozen path resolves outside the repository") from exc
    return path


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ConfigError(f"{name} must be a finite non-negative number")
    return number


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


def _has_failures(checks: Sequence[CheckResult]) -> bool:
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
