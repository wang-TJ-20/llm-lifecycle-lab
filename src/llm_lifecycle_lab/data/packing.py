"""Immutable memory-mapped token packing for native pretraining."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from llm_lifecycle_lab.contracts import (
    SCHEMA_VERSION,
    JsonContract,
    RecordKind,
    utc_now,
)
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.prepare import (
    load_data_manifest,
    verify_data_manifest,
)
from llm_lifecycle_lab.exceptions import (
    ArtifactError,
    ContractError,
    DataValidationError,
)
from llm_lifecycle_lab.tokenizer.native import NativeTokenizer

_TOKEN_DTYPE = np.dtype("<i4")
_LANGUAGE_DTYPE = np.dtype("i1")
_BYTE_WEIGHT_DTYPE = np.dtype("<f4")
LANGUAGE_IDS = {"en": 0, "zh": 1}


@dataclass(frozen=True, slots=True)
class PackingStats:
    documents: int
    source_tokens: int
    supervised_tokens: int
    source_bytes: int
    examples: int
    padding_tokens: int
    sequence_length: int


@dataclass(frozen=True, slots=True)
class PackedArrayManifest(JsonContract):
    path: str
    sha256: str
    size_bytes: int
    dtype: str
    length: int

    def __post_init__(self) -> None:
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ContractError("packed array path must remain inside its directory")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ContractError("packed array sha256 must be lowercase hexadecimal")
        if self.size_bytes < 0 or self.length < 0:
            raise ContractError("packed array size and length must be non-negative")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ContractError(f"invalid packed array dtype: {self.dtype}") from exc
        if self.size_bytes != self.length * dtype.itemsize:
            raise ContractError("packed array byte size does not match its dtype")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackedArrayManifest:
        return cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            size_bytes=int(data["size_bytes"]),
            dtype=str(data["dtype"]),
            length=int(data["length"]),
        )


@dataclass(frozen=True, slots=True)
class PackedSplitManifest(JsonContract):
    name: str
    documents: int
    source_tokens: int
    supervised_tokens: int
    source_bytes: int
    examples: int
    padding_tokens: int
    token_ids: PackedArrayManifest
    language_ids: PackedArrayManifest
    byte_weights: PackedArrayManifest

    def __post_init__(self) -> None:
        if self.source_tokens < 2 or self.supervised_tokens != self.source_tokens - 1:
            raise ContractError("packed split token counts are inconsistent")
        if self.documents <= 0 or self.examples <= 0:
            raise ContractError("packed split must contain documents and examples")
        if self.source_bytes < 0 or self.padding_tokens < 0:
            raise ContractError(
                "packed split byte and padding counts must be non-negative"
            )
        arrays = (self.token_ids, self.language_ids, self.byte_weights)
        if any(array.length != self.source_tokens for array in arrays):
            raise ContractError("packed split array lengths are inconsistent")
        if (
            self.token_ids.dtype != _TOKEN_DTYPE.str
            or self.language_ids.dtype != _LANGUAGE_DTYPE.str
            or self.byte_weights.dtype != _BYTE_WEIGHT_DTYPE.str
        ):
            raise ContractError("packed split array dtypes are unsupported")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackedSplitManifest:
        return cls(
            name=str(data["name"]),
            documents=int(data["documents"]),
            source_tokens=int(data["source_tokens"]),
            supervised_tokens=int(data["supervised_tokens"]),
            source_bytes=int(data["source_bytes"]),
            examples=int(data["examples"]),
            padding_tokens=int(data["padding_tokens"]),
            token_ids=PackedArrayManifest.from_dict(data["token_ids"]),
            language_ids=PackedArrayManifest.from_dict(data["language_ids"]),
            byte_weights=PackedArrayManifest.from_dict(data["byte_weights"]),
        )


@dataclass(frozen=True, slots=True)
class PackedPretrainingManifest(JsonContract):
    dataset_id: str
    data_manifest_sha256: str
    tokenizer_sha256: str
    sequence_length: int
    pad_token_id: int
    splits: tuple[PackedSplitManifest, ...]
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ContractError("packed dataset_id must not be empty")
        for name, digest in (
            ("data_manifest_sha256", self.data_manifest_sha256),
            ("tokenizer_sha256", self.tokenizer_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ContractError(f"packed {name} must be lowercase hexadecimal")
        if self.sequence_length < 2:
            raise ContractError("packed sequence_length must be at least 2")
        if self.pad_token_id < 0:
            raise ContractError("packed pad_token_id must be non-negative")
        if len(self.splits) != 3 or {split.name for split in self.splits} != {
            "train",
            "dev",
            "test",
        }:
            raise ContractError("packed manifest requires train, dev, and test")
        array_paths = [
            array.path
            for split in self.splits
            for array in (
                split.token_ids,
                split.language_ids,
                split.byte_weights,
            )
        ]
        if len(array_paths) != len(set(array_paths)):
            raise ContractError("packed array paths must be unique")
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(
                f"unsupported packed manifest schema_version: {self.schema_version}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackedPretrainingManifest:
        return cls(
            dataset_id=str(data["dataset_id"]),
            data_manifest_sha256=str(data["data_manifest_sha256"]),
            tokenizer_sha256=str(data["tokenizer_sha256"]),
            sequence_length=int(data["sequence_length"]),
            pad_token_id=int(data["pad_token_id"]),
            splits=tuple(
                PackedSplitManifest.from_dict(item) for item in data["splits"]
            ),
            created_at=str(data["created_at"]),
            schema_version=str(data["schema_version"]),
        )


class DiskPackedPretrainingDataset(Dataset[dict[str, Tensor]]):
    """Read fixed-length examples from immutable memory-mapped token arrays."""

    def __init__(
        self,
        root: Path,
        split: PackedSplitManifest,
        *,
        sequence_length: int,
        pad_token_id: int,
    ) -> None:
        self.root = root
        self.split = split
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self._token_ids = np.memmap(
            root / split.token_ids.path,
            mode="r",
            dtype=_TOKEN_DTYPE,
            shape=(split.source_tokens,),
        )
        self._language_ids = np.memmap(
            root / split.language_ids.path,
            mode="r",
            dtype=_LANGUAGE_DTYPE,
            shape=(split.source_tokens,),
        )
        self._byte_weights = np.memmap(
            root / split.byte_weights.path,
            mode="r",
            dtype=_BYTE_WEIGHT_DTYPE,
            shape=(split.source_tokens,),
        )
        self.stats = PackingStats(
            documents=split.documents,
            source_tokens=split.source_tokens,
            supervised_tokens=split.supervised_tokens,
            source_bytes=split.source_bytes,
            examples=split.examples,
            padding_tokens=split.padding_tokens,
            sequence_length=sequence_length,
        )

    @classmethod
    def from_manifest(
        cls,
        packed_manifest_path: str | Path,
        *,
        split: str,
        data_manifest_sha256: str,
        tokenizer_sha256: str,
        sequence_length: int,
    ) -> DiskPackedPretrainingDataset:
        path = Path(packed_manifest_path)
        manifest = load_packed_pretraining_manifest(path)
        failures = verify_packed_pretraining_manifest(path)
        if failures:
            raise DataValidationError("; ".join(failures))
        expected = {
            "data manifest": (
                manifest.data_manifest_sha256,
                data_manifest_sha256,
            ),
            "tokenizer": (manifest.tokenizer_sha256, tokenizer_sha256),
            "sequence length": (manifest.sequence_length, sequence_length),
        }
        mismatches = [
            name for name, (actual, wanted) in expected.items() if actual != wanted
        ]
        if mismatches:
            raise DataValidationError(
                "packed pretraining data is incompatible: " + ", ".join(mismatches)
            )
        try:
            split_manifest = next(
                item for item in manifest.splits if item.name == split
            )
        except StopIteration as exc:
            raise DataValidationError(f"unknown packed split: {split}") from exc
        return cls(
            path.parent,
            split_manifest,
            sequence_length=sequence_length,
            pad_token_id=manifest.pad_token_id,
        )

    def __len__(self) -> int:
        return self.split.examples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        stride = self.sequence_length - 1
        start = index * stride
        stop = min(start + self.sequence_length, self.split.source_tokens)
        valid_length = stop - start
        padding = self.sequence_length - valid_length

        token_ids = self._token_ids[start:stop].astype(np.int64).tolist()
        language_ids = self._language_ids[start:stop].astype(np.int8).tolist()
        byte_weights = self._byte_weights[start:stop].astype(np.float32).tolist()
        source_bytes = sum(byte_weights[1:])
        return {
            "input_ids": torch.tensor(
                token_ids + [self.pad_token_id] * padding,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [1] * valid_length + [0] * padding,
                dtype=torch.bool,
            ),
            "labels": torch.tensor(
                token_ids + [-100] * padding,
                dtype=torch.long,
            ),
            "language_ids": torch.tensor(
                language_ids + [-1] * padding,
                dtype=torch.int8,
            ),
            "byte_weights": torch.tensor(
                byte_weights + [0.0] * padding,
                dtype=torch.float32,
            ),
            "source_bytes": torch.tensor(source_bytes, dtype=torch.float32),
        }


def collate_pretraining_batch(
    examples: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    if not examples:
        raise DataValidationError("cannot collate an empty batch")
    keys = (
        "input_ids",
        "attention_mask",
        "labels",
        "language_ids",
        "byte_weights",
        "source_bytes",
    )
    batch = {key: torch.stack([example[key] for example in examples]) for key in keys}
    # Check on CPU before transfer, avoiding a device synchronization in each layer.
    mask = batch["attention_mask"]
    if mask.device.type == "cpu" and bool(mask.all()):
        del batch["attention_mask"]
    return batch


def materialize_packed_pretraining_dataset(
    data_manifest_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    sequence_length: int,
) -> PackedPretrainingManifest:
    if sequence_length < 2:
        raise DataValidationError("sequence_length must be at least 2")
    data_path = Path(data_manifest_path)
    failures = verify_data_manifest(data_path)
    if failures:
        raise DataValidationError("; ".join(failures))
    data_manifest = load_data_manifest(data_path)
    if data_manifest.record_kind is not RecordKind.PRETRAIN:
        raise DataValidationError("token packing requires pretrain data")
    tokenizer = NativeTokenizer.from_directory(tokenizer_path)
    data_sha256 = sha256_file(data_path)
    if tokenizer.manifest.source_data_sha256 != data_sha256:
        raise DataValidationError(
            "tokenizer was trained from a different Data Manifest"
        )

    target = Path(output_dir)
    if target.exists():
        raise ArtifactError(
            f"packed output already exists; refusing to overwrite: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
    )
    try:
        splits = tuple(
            _materialize_split(
                data_path.parent / split.path,
                split.name,
                temporary,
                tokenizer=tokenizer,
                sequence_length=sequence_length,
            )
            for split in data_manifest.splits
        )
        manifest = PackedPretrainingManifest(
            dataset_id=data_manifest.dataset_id,
            data_manifest_sha256=data_sha256,
            tokenizer_sha256=tokenizer.manifest.content_sha256,
            sequence_length=sequence_length,
            pad_token_id=tokenizer.pad_token_id,
            splits=splits,
        )
        _write_json(temporary / "packed_manifest.json", manifest.to_dict())
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_packed_pretraining_manifest(
    path: str | Path,
) -> PackedPretrainingManifest:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("packed manifest root must be an object")
        return PackedPretrainingManifest.from_dict(value)
    except (OSError, KeyError, TypeError, ValueError, ContractError) as exc:
        raise DataValidationError(
            f"invalid packed pretraining manifest {manifest_path}: {exc}"
        ) from exc


def verify_packed_pretraining_manifest(path: str | Path) -> list[str]:
    manifest_path = Path(path)
    manifest = load_packed_pretraining_manifest(manifest_path)
    root = manifest_path.parent.resolve()
    failures: list[str] = []
    for split in manifest.splits:
        for array in (
            split.token_ids,
            split.language_ids,
            split.byte_weights,
        ):
            array_path = manifest_path.parent / array.path
            if array_path.is_symlink() or not array_path.resolve().is_relative_to(root):
                failures.append(
                    f"packed array must be a regular file inside its directory: "
                    f"{array.path}"
                )
                continue
            if not array_path.is_file():
                failures.append(f"missing packed array: {array.path}")
                continue
            if array_path.stat().st_size != array.size_bytes:
                failures.append(f"packed array size mismatch: {array.path}")
            if sha256_file(array_path) != array.sha256:
                failures.append(f"packed array hash mismatch: {array.path}")
    return failures


def _materialize_split(
    source_path: Path,
    split_name: str,
    output_dir: Path,
    *,
    tokenizer: NativeTokenizer,
    sequence_length: int,
) -> PackedSplitManifest:
    token_path = output_dir / f"{split_name}.tokens.i32"
    language_path = output_dir / f"{split_name}.languages.i8"
    byte_weight_path = output_dir / f"{split_name}.byte_weights.f32"
    documents = 0
    source_tokens = 0
    source_bytes = 0

    try:
        with (
            source_path.open("r", encoding="utf-8") as source,
            token_path.open("wb") as token_file,
            language_path.open("wb") as language_file,
            byte_weight_path.open("wb") as byte_weight_file,
        ):
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                text = value.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise DataValidationError(
                        f"{source_path} line {line_number} has no text"
                    )
                content_tokens = tokenizer.encode(text)
                document_tokens = np.asarray(
                    [
                        tokenizer.bos_token_id,
                        *content_tokens,
                        tokenizer.eos_token_id,
                    ],
                    dtype=_TOKEN_DTYPE,
                )
                normalized_bytes = len(tokenizer.decode(content_tokens).encode("utf-8"))
                language_id = LANGUAGE_IDS.get(value.get("language"), -1)
                languages = np.full(
                    len(document_tokens),
                    language_id,
                    dtype=_LANGUAGE_DTYPE,
                )
                byte_weights = np.zeros(
                    len(document_tokens),
                    dtype=_BYTE_WEIGHT_DTYPE,
                )
                if content_tokens:
                    byte_weights[1:-1] = normalized_bytes / len(content_tokens)

                document_tokens.tofile(token_file)
                languages.tofile(language_file)
                byte_weights.tofile(byte_weight_file)
                documents += 1
                source_tokens += len(document_tokens)
                source_bytes += normalized_bytes
    except json.JSONDecodeError as exc:
        raise DataValidationError(
            f"invalid JSON in prepared split {source_path}"
        ) from exc
    except OSError as exc:
        raise DataValidationError(
            f"cannot materialize packed split {source_path}: {exc}"
        ) from exc

    if source_tokens < 2:
        raise DataValidationError(
            f"packed split {split_name} has fewer than two tokens"
        )
    stride = sequence_length - 1
    examples = math.ceil((source_tokens - 1) / stride)
    final_valid_length = source_tokens - (examples - 1) * stride
    padding_tokens = sequence_length - final_valid_length
    return PackedSplitManifest(
        name=split_name,
        documents=documents,
        source_tokens=source_tokens,
        supervised_tokens=source_tokens - 1,
        source_bytes=source_bytes,
        examples=examples,
        padding_tokens=padding_tokens,
        token_ids=_array_manifest(token_path, _TOKEN_DTYPE, source_tokens),
        language_ids=_array_manifest(
            language_path,
            _LANGUAGE_DTYPE,
            source_tokens,
        ),
        byte_weights=_array_manifest(
            byte_weight_path,
            _BYTE_WEIGHT_DTYPE,
            source_tokens,
        ),
    )


def _array_manifest(
    path: Path,
    dtype: np.dtype[Any],
    length: int,
) -> PackedArrayManifest:
    return PackedArrayManifest(
        path=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        dtype=dtype.str,
        length=length,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise DataValidationError(
            f"cannot write packed manifest {path}: {exc}"
        ) from exc
