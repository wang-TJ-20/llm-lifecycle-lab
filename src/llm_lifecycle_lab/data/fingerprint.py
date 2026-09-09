"""Content fingerprints used by data and artifact manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.contracts import FileFingerprint
from llm_lifecycle_lab.exceptions import DataValidationError

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise DataValidationError(f"file does not exist: {file_path}")

    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DataValidationError(f"cannot read {file_path}: {exc}") from exc
    return digest.hexdigest()


def fingerprint_file(
    path: str | Path,
    *,
    display_path: str | None = None,
) -> FileFingerprint:
    file_path = Path(path)
    try:
        size = file_path.stat().st_size
    except OSError as exc:
        raise DataValidationError(f"cannot stat {file_path}: {exc}") from exc
    return FileFingerprint(
        path=display_path or str(file_path),
        sha256=sha256_file(file_path),
        size_bytes=size,
    )


def canonical_record_sha256(record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
