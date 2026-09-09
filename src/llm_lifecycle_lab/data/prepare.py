"""Create versioned, validated dataset splits and a manifest."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.contracts import (
    DataManifest,
    DataSplitManifest,
    RecordKind,
)
from llm_lifecycle_lab.data.fingerprint import fingerprint_file, sha256_file
from llm_lifecycle_lab.data.schemas import (
    format_validation_failure,
    validate_jsonl,
)
from llm_lifecycle_lab.data.split import (
    SplitRatios,
    group_value,
    split_records,
)
from llm_lifecycle_lab.exceptions import ContractError, DataValidationError

_DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def prepare_dataset(
    source: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str,
    record_kind: RecordKind | str,
    license_name: str,
    seed: int = 42,
    group_by: str = "id",
    ratios: SplitRatios | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> DataManifest:
    """Validate, split, fingerprint, and atomically publish a dataset."""

    if not _DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise DataValidationError(
            "dataset_id must contain only letters, numbers, '.', '_' or '-' "
            "and be at most 128 characters"
        )
    if not license_name.strip():
        raise DataValidationError("license_name must not be empty")

    source_path = Path(source)
    target = Path(output_dir)
    if target.exists():
        raise DataValidationError(
            f"output directory already exists; refusing to overwrite: {target}"
        )

    validated = validate_jsonl(source_path, record_kind)
    if not validated.report.ok:
        raise DataValidationError(format_validation_failure(validated.report))

    selected_ratios = ratios or SplitRatios()
    splits = split_records(
        validated.records,
        ratios=selected_ratios,
        seed=seed,
        group_by=group_by,
    )

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
            )
        )
    except OSError as exc:
        raise DataValidationError(
            f"cannot create temporary dataset directory for {target}: {exc}"
        ) from exc

    try:
        split_manifests = []
        for split_name in ("train", "dev", "test"):
            split_path = temporary / f"{split_name}.jsonl"
            records = splits[split_name]
            _write_jsonl(split_path, records)
            split_manifests.append(
                DataSplitManifest(
                    name=split_name,
                    path=split_path.name,
                    sha256=sha256_file(split_path),
                    records=len(records),
                    groups=len({group_value(record, group_by) for record in records}),
                )
            )

        manifest = DataManifest(
            dataset_id=dataset_id,
            record_kind=RecordKind(record_kind),
            source=fingerprint_file(
                source_path,
                display_path=str(source_path.resolve()),
            ),
            splits=tuple(split_manifests),
            split_seed=seed,
            group_by=group_by,
            license=license_name,
            processing_steps=(
                "validate-schema",
                f"deterministic-group-split:{selected_ratios.train}:"
                f"{selected_ratios.dev}:{selected_ratios.test}",
            ),
            source_metadata=source_metadata or {},
        )
        _write_json(temporary / "data_manifest.json", manifest.to_dict())
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return manifest


def load_data_manifest(path: str | Path) -> DataManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise DataValidationError(f"data manifest does not exist: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise DataValidationError("data manifest root must be an object")
        return DataManifest.from_dict(value)
    except (
        OSError,
        TypeError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
        ContractError,
    ) as exc:
        if isinstance(exc, DataValidationError):
            raise
        raise DataValidationError(
            f"invalid data manifest {manifest_path}: {exc}"
        ) from exc


def verify_data_manifest(path: str | Path) -> list[str]:
    """Return deterministic verification failures for prepared split files."""

    manifest_path = Path(path)
    manifest = load_data_manifest(manifest_path)
    failures: list[str] = []

    for split in manifest.splits:
        split_path = manifest_path.parent / split.path
        if not split_path.is_file():
            failures.append(f"missing split file: {split.path}")
            continue
        actual_sha256 = sha256_file(split_path)
        if actual_sha256 != split.sha256:
            failures.append(
                f"hash mismatch for {split.path}: "
                f"expected {split.sha256}, got {actual_sha256}"
            )
        actual_records = _count_jsonl_records(split_path)
        if actual_records != split.records:
            failures.append(
                f"record count mismatch for {split.path}: "
                f"expected {split.records}, got {actual_records}"
            )
    return failures


def _write_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            for record in records:
                line = json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write(f"{line}\n")
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, TypeError, ValueError) as exc:
        raise DataValidationError(f"cannot write split {path}: {exc}") from exc


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                dict(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, TypeError, ValueError) as exc:
        raise DataValidationError(f"cannot write manifest {path}: {exc}") from exc


def _count_jsonl_records(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError as exc:
        raise DataValidationError(f"cannot read split {path}: {exc}") from exc
