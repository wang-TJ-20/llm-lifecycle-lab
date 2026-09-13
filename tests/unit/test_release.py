from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from llm_lifecycle_lab.exceptions import ArtifactError
from llm_lifecycle_lab.release import (
    EXPECTED_CHECKPOINT_FILES,
    extract_checkpoint_archive,
    verify_release_files,
    write_release_manifest,
)


def _write_checkpoint_archive(path: Path, *, unsafe: bool = False) -> None:
    with tarfile.open(path, mode="w:gz") as handle:
        names = list(EXPECTED_CHECKPOINT_FILES)
        if unsafe:
            names.append("../outside")
        for name in names:
            payload = name.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))


def test_extract_checkpoint_archive_requires_exact_safe_payload(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "checkpoint.tar.gz"
    _write_checkpoint_archive(archive)

    target = extract_checkpoint_archive(archive, tmp_path / "checkpoint")

    assert {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    } == EXPECTED_CHECKPOINT_FILES
    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        extract_checkpoint_archive(archive, target)


def test_extract_checkpoint_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _write_checkpoint_archive(archive, unsafe=True)

    with pytest.raises(ArtifactError, match="unsafe path"):
        extract_checkpoint_archive(archive, tmp_path / "checkpoint")

    assert not (tmp_path / "outside").exists()


def test_release_manifest_detects_tampering(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    payload = release / "model.bin"
    payload.write_bytes(b"weights")
    write_release_manifest(
        release,
        {
            "schema_version": "1.0",
            "model_id": "test-model",
            "repository_id": "test/test-model",
        },
    )

    result = verify_release_files(release)

    assert result["files_verified"] == 1
    assert result["sha256sums_verified"] == 2
    payload.write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="SHA-256 mismatch"):
        verify_release_files(release)
