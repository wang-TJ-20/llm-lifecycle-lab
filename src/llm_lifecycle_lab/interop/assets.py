"""Version gates and content-addressed local HF artifact manifests."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.artifacts import RunArtifacts
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.evaluation.suite import read_json
from llm_lifecycle_lab.exceptions import ArtifactError

TRANSFORMERS_VERSION = "4.53.3"
PEFT_VERSION = "0.17.1"


def require_hf(*, peft: bool = False) -> Any:
    expected = {"transformers": TRANSFORMERS_VERSION}
    if peft:
        expected["peft"] = PEFT_VERSION
    for package, version in expected.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ArtifactError(
                "Install optional HF dependencies: "
                "python -m pip install -r requirements-hf.txt"
            ) from exc
        if actual != version:
            raise ArtifactError(f"{package} must be {version}, found {actual}")
    return importlib.import_module("transformers")


def payload_fingerprints(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ArtifactError(f"HF artifacts cannot contain symlinks: {path}")
        if path.is_file() and path.name != "hf_manifest.json":
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return files


def write_hf_manifest(root: Path, metadata: dict[str, Any]) -> Path:
    return RunArtifacts(root.name, root).write_json(
        "hf_manifest.json",
        {"schema_version": "1.0", **metadata, "files": payload_fingerprints(root)},
    )


def verify_hf_assets(root: str | Path) -> dict[str, Any]:
    directory = Path(root)
    manifest = read_json(directory / "hf_manifest.json")
    if manifest.get("schema_version") != "1.0" or not manifest.get("files"):
        raise ArtifactError("invalid HF artifact manifest")
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactError("HF manifest path escapes artifact directory")
    # Exact payload equality prevents unrecorded configs or code from being loaded.
    if payload_fingerprints(directory) != manifest["files"]:
        raise ArtifactError("HF artifact content/hash mismatch")
    return manifest
