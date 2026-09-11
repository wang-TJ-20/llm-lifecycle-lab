"""Capture runtime and source provenance for reproducible training runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab import __version__
from llm_lifecycle_lab.config import canonical_json
from llm_lifecycle_lab.contracts import SCHEMA_VERSION, utc_now
from llm_lifecycle_lab.data.fingerprint import sha256_file

_PACKAGES = (
    "llm-lifecycle-lab",
    "numpy",
    "PyYAML",
    "tokenizers",
    "torch",
)


def capture_runtime_provenance(
    *,
    workdir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Return a JSON-safe snapshot without recording secrets or full env vars."""

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "project_version": __version__,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {package: _package_version(package) for package in _PACKAGES},
        "accelerator": _accelerator_details(device),
        "code": _git_details(Path(workdir)),
    }


def source_tree_sha256(workdir: str | Path) -> str | None:
    """Hash Python source contents, independent of checkout path and timestamps."""
    root = Path(workdir).resolve()
    files = sorted((root / "src/llm_lifecycle_lab").rglob("*.py"))
    if not files:
        return None
    contents = {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
    return hashlib.sha256(canonical_json(contents).encode("utf-8")).hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _accelerator_details(device: torch.device) -> dict[str, Any]:
    details: dict[str, Any] = {
        "type": device.type,
        "selected_device": str(device),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "devices": [],
    }
    if device.type == "cuda":
        details["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_properties(index).name,
                "total_memory_bytes": int(
                    torch.cuda.get_device_properties(index).total_memory
                ),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    elif device.type == "mps":
        details["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    return details


def _git_details(workdir: Path) -> dict[str, Any]:
    commit = _run_git(workdir, "rev-parse", "--verify", "HEAD")
    status = _run_git(workdir, "status", "--porcelain=v1")
    return {
        "commit": commit,
        "branch": _run_git(workdir, "branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "status_entries": (None if status is None else len(status.splitlines())),
        "source_sha256": source_tree_sha256(workdir),
    }


def _run_git(workdir: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env={
                **os.environ,
                "LC_ALL": "C",
            },
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
