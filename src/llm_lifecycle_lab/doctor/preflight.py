"""Orchestrate deterministic preflight checks before expensive work."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from llm_lifecycle_lab.config import load_mapping
from llm_lifecycle_lab.contracts import ModelRoute, RunConfig, RunProfile, Stage
from llm_lifecycle_lab.data.prepare import (
    load_data_manifest,
    verify_data_manifest,
)
from llm_lifecycle_lab.doctor.environment import (
    check_disk_space,
    check_package,
    check_python,
    check_write_access,
    environment_details,
)
from llm_lifecycle_lab.doctor.hardware import check_torch_and_device
from llm_lifecycle_lab.doctor.result import (
    CheckResult,
    CheckStatus,
    DoctorReport,
)
from llm_lifecycle_lab.doctor.runtime import check_real_batch
from llm_lifecycle_lab.exceptions import ConfigError, DataValidationError


@dataclass(frozen=True, slots=True)
class ProfileRequirements:
    require_torch: bool
    require_cuda: bool
    minimum_cuda_gib: float
    minimum_disk_gib: float
    require_linux: bool
    require_transformers: bool = False


_PROFILES = {
    "base": ProfileRequirements(
        require_torch=False,
        require_cuda=False,
        minimum_cuda_gib=0,
        minimum_disk_gib=1,
        require_linux=False,
    ),
    "smoke-10m": ProfileRequirements(
        require_torch=True,
        require_cuda=False,
        minimum_cuda_gib=0,
        minimum_disk_gib=2,
        require_linux=False,
    ),
    "tiny-60m-local": ProfileRequirements(
        require_torch=True,
        require_cuda=False,
        minimum_cuda_gib=0,
        minimum_disk_gib=5,
        require_linux=False,
    ),
    "tiny-60m": ProfileRequirements(
        require_torch=True,
        require_cuda=True,
        minimum_cuda_gib=22,
        minimum_disk_gib=20,
        require_linux=True,
    ),
    "qwen3-0.6b-base": ProfileRequirements(
        require_torch=True,
        require_cuda=True,
        minimum_cuda_gib=22,
        minimum_disk_gib=30,
        require_linux=True,
        require_transformers=True,
    ),
}


def available_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES)


def profile_for_config(config: RunConfig) -> str:
    if config.model_route is ModelRoute.QWEN3_TRANSFER:
        return "qwen3-0.6b-base"
    if config.run_profile is RunProfile.SMOKE:
        if str(config.model.get("model_id", "")).startswith("tiny-60m"):
            return "tiny-60m-local"
        return "smoke-10m"
    return "tiny-60m"


def run_doctor(
    *,
    profile: str | None = None,
    config: RunConfig | None = None,
    workdir: str | Path = ".",
) -> DoctorReport:
    selected_profile = profile or (
        profile_for_config(config) if config is not None else "base"
    )
    if selected_profile not in _PROFILES:
        choices = ", ".join(available_profiles())
        raise ConfigError(
            f"unknown doctor profile {selected_profile!r}; choose one of {choices}"
        )

    requirements = _PROFILES[selected_profile]
    root = Path(workdir).resolve()
    output_path = root / config.output_dir if config is not None else root / "runs"
    checks = [
        CheckResult(
            name="environment",
            status=CheckStatus.PASS,
            message="captured runtime environment",
            details=environment_details(),
        ),
        check_python(),
        check_package("PyYAML", required=True),
        check_write_access(output_path),
        check_disk_space(
            output_path,
            minimum_gib=requirements.minimum_disk_gib,
        ),
    ]

    if config is not None:
        checks.append(_check_profile_matches_config(selected_profile, config))
        checks.extend(_check_model_config(config, root))
        checks.extend(_check_data(config, root))

    if requirements.require_linux:
        is_linux = platform.system() == "Linux"
        checks.append(
            CheckResult(
                name="operating-system",
                status=CheckStatus.PASS if is_linux else CheckStatus.FAIL,
                message=(
                    "Linux is available"
                    if is_linux
                    else f"{selected_profile} is supported only on Linux"
                ),
                details={"platform": platform.system()},
            )
        )

    torch_check, device_check = check_torch_and_device(
        require_torch=requirements.require_torch,
        require_cuda=requirements.require_cuda,
        minimum_cuda_gib=requirements.minimum_cuda_gib,
    )
    checks.extend((torch_check, device_check))

    if requirements.require_transformers:
        exact_version = None
        if config is not None and config.model.get("transformers_version"):
            exact_version = str(config.model["transformers_version"])
        checks.append(
            check_package(
                "transformers",
                required=True,
                exact_version=exact_version,
            )
        )

    if config is not None:
        checks.append(check_real_batch(config, root))

    return DoctorReport(profile=selected_profile, checks=tuple(checks))


def _check_profile_matches_config(
    selected_profile: str,
    config: RunConfig,
) -> CheckResult:
    expected = profile_for_config(config)
    matches = selected_profile == expected
    return CheckResult(
        name="profile-config-match",
        status=CheckStatus.PASS if matches else CheckStatus.FAIL,
        message=(
            f"profile matches config run profile: {expected}"
            if matches
            else f"profile {selected_profile} conflicts with config run profile; "
            f"expected {expected}"
        ),
        details={
            "selected_profile": selected_profile,
            "expected_profile": expected,
            "model_route": config.model_route.value,
            "run_profile": config.run_profile.value,
        },
    )


def _check_data(config: RunConfig, root: Path) -> list[CheckResult]:
    manifest_value = config.data.get("manifest")
    if not manifest_value:
        return [
            CheckResult(
                name="data-manifest",
                status=CheckStatus.WARN,
                message="config does not declare data.manifest",
            )
        ]

    manifest_path = Path(str(manifest_value))
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    try:
        failures = verify_data_manifest(manifest_path)
    except DataValidationError as exc:
        return [
            CheckResult(
                name="data-manifest",
                status=CheckStatus.FAIL,
                message=str(exc),
                details={"path": str(manifest_path)},
            )
        ]

    if failures:
        return [
            CheckResult(
                name="data-manifest",
                status=CheckStatus.FAIL,
                message="; ".join(failures),
                details={"path": str(manifest_path)},
            )
        ]
    manifest = load_data_manifest(manifest_path)
    if config.stage is Stage.PRETRAIN and any(
        split.name == "dev" and split.records == 0 for split in manifest.splits
    ):
        return [
            CheckResult(
                name="data-manifest",
                status=CheckStatus.FAIL,
                message="pretraining requires a non-empty dev split",
                details={"path": str(manifest_path)},
            )
        ]
    checks = [
        CheckResult(
            name="data-manifest",
            status=CheckStatus.PASS,
            message=f"data manifest verified: {manifest_path}",
            details={"path": str(manifest_path)},
        )
    ]
    packed_value = config.data.get("packed_manifest")
    if not packed_value:
        if config.stage is Stage.PRETRAIN:
            checks.append(
                CheckResult(
                    name="packed-data",
                    status=CheckStatus.FAIL,
                    message="pretraining config must declare data.packed_manifest",
                )
            )
        return checks
    packed_path = Path(str(packed_value))
    if not packed_path.is_absolute():
        packed_path = root / packed_path
    try:
        from llm_lifecycle_lab.data.fingerprint import sha256_file
        from llm_lifecycle_lab.data.packing import (
            load_packed_pretraining_manifest,
            verify_packed_pretraining_manifest,
        )

        packed_failures = verify_packed_pretraining_manifest(packed_path)
        packed_manifest = load_packed_pretraining_manifest(packed_path)
        if packed_manifest.data_manifest_sha256 != sha256_file(manifest_path):
            packed_failures.append("packed data manifest hash mismatch")
        tokenizer_value = config.model.get("tokenizer")
        if not tokenizer_value:
            packed_failures.append(
                "config does not declare model.tokenizer for packed data"
            )
        else:
            tokenizer_path = Path(str(tokenizer_value))
            if not tokenizer_path.is_absolute():
                tokenizer_path = root / tokenizer_path
            tokenizer_file = tokenizer_path / "tokenizer.json"
            if not tokenizer_file.is_file():
                packed_failures.append(
                    f"tokenizer file does not exist: {tokenizer_file}"
                )
            elif packed_manifest.tokenizer_sha256 != sha256_file(tokenizer_file):
                packed_failures.append("packed tokenizer hash mismatch")
        sequence_length = config.training.get("sequence_length")
        if (
            not isinstance(sequence_length, int)
            or isinstance(sequence_length, bool)
            or sequence_length < 2
        ):
            packed_failures.append(
                "config does not declare a valid training.sequence_length"
            )
        elif packed_manifest.sequence_length != sequence_length:
            packed_failures.append("packed sequence length mismatch")
    except (ImportError, DataValidationError) as exc:
        packed_failures = [str(exc)]
    checks.append(
        CheckResult(
            name="packed-data",
            status=(CheckStatus.FAIL if packed_failures else CheckStatus.PASS),
            message=(
                "; ".join(packed_failures)
                if packed_failures
                else f"packed data verified: {packed_path}"
            ),
            details={"path": str(packed_path)},
        )
    )
    return checks


def _check_model_config(
    config: RunConfig,
    root: Path,
) -> list[CheckResult]:
    model_config_value = config.model.get("config")
    if not model_config_value:
        return [
            CheckResult(
                name="model-config",
                status=CheckStatus.WARN,
                message="config does not declare model.config",
            )
        ]

    model_config_path = Path(str(model_config_value))
    if not model_config_path.is_absolute():
        model_config_path = root / model_config_path
    try:
        model_config = load_mapping(model_config_path)
    except ConfigError as exc:
        return [
            CheckResult(
                name="model-config",
                status=CheckStatus.FAIL,
                message=str(exc),
                details={"path": str(model_config_path)},
            )
        ]

    expected_values = {
        "model_route": config.model_route.value,
        "provider": config.model.get("provider"),
        "model_id": config.model.get("model_id"),
    }
    for optional_field in (
        "revision",
        "tokenizer_revision",
        "transformers_version",
    ):
        if optional_field in config.model:
            expected_values[optional_field] = config.model[optional_field]

    mismatches = [
        f"{field}: expected {expected!r}, got {model_config.get(field)!r}"
        for field, expected in expected_values.items()
        if model_config.get(field) != expected
    ]
    if mismatches:
        return [
            CheckResult(
                name="model-config",
                status=CheckStatus.FAIL,
                message="model config mismatch: " + "; ".join(mismatches),
                details={"path": str(model_config_path)},
            )
        ]
    return [
        CheckResult(
            name="model-config",
            status=CheckStatus.PASS,
            message=f"model config matches pipeline: {model_config_path}",
            details={"path": str(model_config_path)},
        )
    ]
