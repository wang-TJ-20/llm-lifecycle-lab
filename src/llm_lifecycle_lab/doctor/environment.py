"""Environment checks that do not require a model to be loaded."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

from llm_lifecycle_lab.doctor.result import CheckResult, CheckStatus


def check_python(
    minimum: tuple[int, int] = (3, 11),
) -> CheckResult:
    current = sys.version_info[:2]
    status = CheckStatus.PASS if current >= minimum else CheckStatus.FAIL
    return CheckResult(
        name="python",
        status=status,
        message=(
            f"Python {current[0]}.{current[1]} "
            f"{'meets' if status is CheckStatus.PASS else 'does not meet'} "
            f"the >= {minimum[0]}.{minimum[1]} requirement"
        ),
        details={
            "executable": sys.executable,
            "version": platform.python_version(),
            "required": f">={minimum[0]}.{minimum[1]}",
        },
    )


def check_package(
    distribution: str,
    *,
    required: bool,
    exact_version: str | None = None,
) -> CheckResult:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return CheckResult(
            name=f"package:{distribution}",
            status=CheckStatus.FAIL if required else CheckStatus.WARN,
            message=f"{distribution} is not installed",
            details={"required": required},
        )
    if exact_version is not None and version != exact_version:
        return CheckResult(
            name=f"package:{distribution}",
            status=CheckStatus.FAIL if required else CheckStatus.WARN,
            message=(
                f"{distribution} {version} is installed, but "
                f"{exact_version} is required"
            ),
            details={
                "version": version,
                "required": required,
                "exact_version": exact_version,
            },
        )
    return CheckResult(
        name=f"package:{distribution}",
        status=CheckStatus.PASS,
        message=f"{distribution} {version} is installed",
        details={
            "version": version,
            "required": required,
            "exact_version": exact_version,
        },
    )


def check_write_access(path: str | Path) -> CheckResult:
    target = Path(path)
    candidate = _nearest_existing_parent(target)
    try:
        with tempfile.NamedTemporaryFile(dir=candidate, delete=True):
            pass
    except OSError as exc:
        return CheckResult(
            name="output-write-access",
            status=CheckStatus.FAIL,
            message=f"cannot write under {candidate}: {exc}",
            details={"requested_path": str(target), "checked_path": str(candidate)},
        )
    return CheckResult(
        name="output-write-access",
        status=CheckStatus.PASS,
        message=f"output location is writable: {candidate}",
        details={"requested_path": str(target), "checked_path": str(candidate)},
    )


def check_disk_space(
    path: str | Path,
    *,
    minimum_gib: float,
) -> CheckResult:
    target = _nearest_existing_parent(Path(path))
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return CheckResult(
            name="disk-space",
            status=CheckStatus.FAIL,
            message=f"cannot inspect disk space for {target}: {exc}",
        )

    free_gib = usage.free / 1024**3
    status = CheckStatus.PASS if free_gib >= minimum_gib else CheckStatus.FAIL
    return CheckResult(
        name="disk-space",
        status=status,
        message=(
            f"{free_gib:.1f} GiB free; profile requires at least {minimum_gib:.1f} GiB"
        ),
        details={
            "path": str(target),
            "free_bytes": usage.free,
            "required_bytes": int(minimum_gib * 1024**3),
        },
    )


def environment_details() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": os.getcwd(),
    }


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return candidate
