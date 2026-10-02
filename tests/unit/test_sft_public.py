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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECIPE = PROJECT_ROOT / "configs/data/sft-public-balanced-v1.yaml"


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
    value = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    value["recipe_id"] = "sft-public-fixture"
    value["selection_namespace"] = "fixture-v1"
    value["sources"] = [value["sources"][0]]
    value["output"] = {
        "record_kind": "sft",
        "records": 1,
        "sha256": _record_sha256(_record()),
    }
    value.pop("preparation")
    value.pop("reserved")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_repository_recipe_adapts_only_frozen_sft_sources() -> None:
    raw = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))

    adapted = sft_public._legacy_transform_recipe(raw)

    assert adapted.recipe_id == "public-60m-v3"
    assert [source.source_id for source in adapted.sources] == [
        "oasst1",
        "dolly",
        "hc3-chinese",
        "squad",
        "cmrc2018",
        "msvamp",
    ]
    assert adapted.expected_records["sft"] == 14_799
    assert (
        adapted.expected_source_sha256["sft"]
        == "9ba984310ef0996237908b5843c67d869796fef37e312d2b811f23a647fa772b"
    )


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
