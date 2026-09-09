"""Atomic, path-safe storage for reproducible run artifacts."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.config import config_sha256, dump_yaml
from llm_lifecycle_lab.contracts import JsonContract, RunConfig, RunManifest
from llm_lifecycle_lab.exceptions import ArtifactError

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def new_run_id() -> str:
    """Create a sortable run identifier without embedding experiment meaning."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Access to one run directory with traversal-safe writes."""

    run_id: str
    path: Path

    def artifact_path(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactError(
                f"artifact path must remain inside the run directory: {relative}"
            )
        target = self.path.joinpath(relative)
        resolved_parent = target.parent.resolve()
        try:
            resolved_parent.relative_to(self.path.resolve())
        except ValueError as exc:
            raise ArtifactError(
                f"artifact path escapes the run directory: {relative}"
            ) from exc
        return target

    def write_text(self, relative_path: str | Path, content: str) -> Path:
        target = self.artifact_path(relative_path)
        _atomic_write_text(target, content)
        return target

    def write_json(
        self,
        relative_path: str | Path,
        value: JsonContract | Mapping[str, Any],
    ) -> Path:
        data = value.to_dict() if isinstance(value, JsonContract) else dict(value)
        try:
            content = json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"artifact is not JSON serializable: {exc}") from exc
        return self.write_text(relative_path, f"{content}\n")

    def append_metric(self, metric: Mapping[str, Any]) -> Path:
        target = self.artifact_path("metrics.jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            line = json.dumps(
                dict(metric),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"metric is not JSON serializable: {exc}") from exc
        try:
            with target.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ArtifactError(f"cannot append metric to {target}: {exc}") from exc
        return target

    def update_status(self, status: str) -> Path:
        manifest_path = self.artifact_path("run_manifest.json")
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = RunManifest.from_dict(value)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ArtifactError(
                f"cannot load run manifest {manifest_path}: {exc}"
            ) from exc
        return self.write_json(
            "run_manifest.json",
            replace(manifest, status=status),
        )


class ArtifactStore:
    """Create immutable run roots and initialize required manifests."""

    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def create_run(
        self,
        config: RunConfig,
        *,
        run_id: str | None = None,
    ) -> RunArtifacts:
        selected_id = run_id or new_run_id()
        if not _RUN_ID_PATTERN.fullmatch(selected_id):
            raise ArtifactError(
                "run_id must contain only letters, numbers, '.', '_' or '-' "
                "and be at most 128 characters"
            )

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            run_path = self.root / selected_id
            run_path.mkdir()
        except FileExistsError as exc:
            raise ArtifactError(f"run already exists: {selected_id}") from exc
        except OSError as exc:
            raise ArtifactError(
                f"cannot create run directory under {self.root}: {exc}"
            ) from exc

        artifacts = RunArtifacts(run_id=selected_id, path=run_path)
        digest = config_sha256(config)
        manifest = RunManifest(
            run_id=selected_id,
            config_sha256=digest,
            model_route=config.model_route,
            run_profile=config.run_profile,
            stage=config.stage,
        )

        try:
            artifacts.write_text(
                "resolved_config.yaml",
                dump_yaml(config.to_dict()),
            )
            artifacts.write_json("run_manifest.json", manifest)
            artifacts.write_text("metrics.jsonl", "")
            for directory in ("checkpoints", "evaluations", "export"):
                artifacts.artifact_path(directory).mkdir()
        except Exception:
            _remove_empty_run(run_path)
            raise
        return artifacts

    def open_run(
        self,
        config: RunConfig,
        *,
        run_id: str,
    ) -> RunArtifacts:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ArtifactError(f"invalid run_id: {run_id}")
        run_path = self.root / run_id
        manifest_path = run_path / "run_manifest.json"
        if not manifest_path.is_file():
            raise ArtifactError(f"run does not exist: {run_id}")
        try:
            manifest = RunManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ArtifactError(
                f"cannot load run manifest {manifest_path}: {exc}"
            ) from exc
        expected_sha256 = config_sha256(config)
        if manifest.config_sha256 != expected_sha256:
            raise ArtifactError(
                "resume config does not match the run's resolved config"
            )
        return RunArtifacts(run_id=run_id, path=run_path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ArtifactError(f"cannot write artifact {path}: {exc}") from exc


def _remove_empty_run(path: Path) -> None:
    """Remove only files created while initializing a failed new run."""

    shutil.rmtree(path, ignore_errors=True)
