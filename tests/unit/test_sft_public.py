from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import llm_lifecycle_lab.data.sft_public as sft_public
from llm_lifecycle_lab.contracts import RecordKind
from llm_lifecycle_lab.data.sft_public import (
    load_public_sft_source_manifest,
    materialize_public_sft,
)
from llm_lifecycle_lab.exceptions import DataValidationError


def _record() -> dict[str, Any]:
    return {
        "id": "oasst1:fixture",
        "source_id": "oasst1:fixture",
        "language": "en",
        "messages": [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ],
        "metadata": {
            "source": "fixture",
            "transform": "oasst-ranked-conversation-v1",
        },
    }


def _record_sha256(record: dict[str, Any]) -> str:
    line = (
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _recipe(tmp_path: Path) -> Path:
    value = {
        "schema_version": "1.0",
        "recipe_id": "sft-public-fixture",
        "description": "Synthetic source for materializer tests.",
        "selection_namespace": "fixture-v1",
        "sources": [
            {
                "source_id": "oasst1",
                "provider": "huggingface",
                "repository": "fixture/source",
                "revision": "1" * 40,
                "file": "fixture.jsonl",
                "file_sha256": "2" * 64,
                "format": "jsonl",
                "split": "train",
                "license": "Apache-2.0",
                "transform": "oasst-ranked-conversation-v1",
                "selection": {
                    "strategy": "stable-hash-rank",
                    "records_per_language": 1,
                    "max_messages": 2,
                    "max_characters": 1000,
                    "require_all_assistant_rank_zero": True,
                },
            }
        ],
        "output": {
            "record_kind": "sft",
            "records": 1,
            "sha256": _record_sha256(_record()),
        },
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_recipe_adapter_preserves_sft_source_contract(tmp_path: Path) -> None:
    raw = yaml.safe_load(_recipe(tmp_path).read_text(encoding="utf-8"))

    adapted = sft_public._legacy_transform_recipe(raw)

    assert adapted.recipe_id == "fixture-v1"
    assert [source.source_id for source in adapted.sources] == ["oasst1"]
    assert adapted.sources[0].stages == ("sft",)
    assert adapted.expected_records["sft"] == 1
    assert adapted.expected_source_sha256["sft"] == _record_sha256(_record())


def test_materializer_writes_verifiable_sft_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _recipe(tmp_path)
    upstream = tmp_path / "upstream"
    upstream.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        sft_public,
        "_download_sources",
        lambda _recipe: (
            {"oasst1": upstream},
            {"huggingface_hub": "fixture", "pyarrow": "fixture"},
        ),
    )
    monkeypatch.setattr(
        sft_public,
        "_read_source_rows",
        lambda _path, _source: (),
    )
    monkeypatch.setattr(
        sft_public,
        "_transform_source",
        lambda _rows, _recipe, _source: {RecordKind.SFT: [_record()]},
    )

    output = tmp_path / "materialized"
    manifest = materialize_public_sft(
        recipe,
        output,
        accepted_licenses=["Apache-2.0"],
    )
    loaded = load_public_sft_source_manifest(output / "source.jsonl")

    assert manifest.records == 1
    assert loaded == manifest
    assert manifest.source_sha256 == _record_sha256(_record())
    assert (output / "sft_source_manifest.json").is_file()
    assert not (output / "source_manifest.json").exists()

    with pytest.raises(DataValidationError, match="refusing to overwrite"):
        materialize_public_sft(
            recipe,
            output,
            accepted_licenses=["Apache-2.0"],
        )


def test_materializer_requires_exact_license_acceptance(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="requires exactly"):
        materialize_public_sft(
            _recipe(tmp_path),
            tmp_path / "materialized",
            accepted_licenses=["CC-BY-SA-4.0"],
        )


def test_manifest_loader_detects_source_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _recipe(tmp_path)
    upstream = tmp_path / "upstream"
    upstream.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        sft_public,
        "_download_sources",
        lambda _recipe: ({"oasst1": upstream}, {"loader": "fixture"}),
    )
    monkeypatch.setattr(
        sft_public,
        "_read_source_rows",
        lambda _path, _source: (),
    )
    monkeypatch.setattr(
        sft_public,
        "_transform_source",
        lambda _rows, _recipe, _source: {RecordKind.SFT: [_record()]},
    )
    output = tmp_path / "materialized"
    materialize_public_sft(
        recipe,
        output,
        accepted_licenses=["Apache-2.0"],
    )
    with (output / "source.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(DataValidationError, match="source hash mismatch"):
        load_public_sft_source_manifest(output / "source.jsonl")
