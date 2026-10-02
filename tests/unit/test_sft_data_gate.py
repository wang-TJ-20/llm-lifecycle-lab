from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

import llm_lifecycle_lab.sft_data_gate as sft_data_gate
from llm_lifecycle_lab.contracts import RecordKind
from llm_lifecycle_lab.data.prepare import prepare_dataset
from llm_lifecycle_lab.data.split import SplitRatios, split_records
from llm_lifecycle_lab.doctor.result import CheckResult, CheckStatus
from llm_lifecycle_lab.exceptions import ConfigError, ContractError
from llm_lifecycle_lab.sft_data_gate import (
    SFTDataGateReport,
    verify_sft_data_gate,
)
from llm_lifecycle_lab.tokenizer import NativeTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = PROJECT_ROOT / "configs/gates/native-60m-sft-v1-data.yaml"
RECIPE_PATH = PROJECT_ROOT / "configs/data/sft-public-balanced-v1.yaml"


def _pass(name: str) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.PASS, message="passed")


def _statuses(report: SFTDataGateReport) -> dict[str, CheckStatus]:
    return {check.name: check.status for check in report.checks}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(
    tmp_path: Path,
    mutate_recipe: Callable[[dict[str, Any]], None] | None = None,
    mutate_spec: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    base_gate = tmp_path / "base-gate.yaml"
    base_gate.write_bytes(
        (PROJECT_ROOT / "configs/gates/native-60m-sft-v1-base-init.yaml").read_bytes()
    )
    recipe_value = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    assert isinstance(recipe_value, dict)
    if mutate_recipe is not None:
        mutate_recipe(recipe_value)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(recipe_value, sort_keys=False),
        encoding="utf-8",
    )

    spec_value = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(spec_value, dict)
    spec_value["locks"] = {
        "base_gate": base_gate.name,
        "base_gate_sha256": _sha256(base_gate),
        "recipe": recipe.name,
        "recipe_sha256": _sha256(recipe),
    }
    if mutate_spec is not None:
        mutate_spec(spec_value)
    spec = tmp_path / "gate.yaml"
    spec.write_text(
        yaml.safe_dump(spec_value, sort_keys=False),
        encoding="utf-8",
    )
    return spec


def test_repository_contract_only_passes_without_materialized_data() -> None:
    report = verify_sft_data_gate(SPEC_PATH, workdir=PROJECT_ROOT)

    assert report.exit_code == 0
    assert report.scope == "contract-only"
    assert report.counts == {"pass": 5, "warn": 0, "fail": 0}
    assert "canonical-source" not in _statuses(report)


def test_inputs_only_fails_when_materialized_data_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _write_fixture(tmp_path)
    monkeypatch.setattr(
        sft_data_gate,
        "_base_initialization_check",
        lambda _spec, _root: _pass("base-initialization"),
    )
    monkeypatch.setattr(
        sft_data_gate,
        "_load_tokenizer",
        lambda _spec: (object(), _pass("tokenizer-identity")),
    )

    report = verify_sft_data_gate(
        spec,
        workdir=tmp_path,
        check_inputs=True,
    )

    assert report.exit_code == 1
    statuses = _statuses(report)
    assert statuses["canonical-source"] is CheckStatus.FAIL
    assert statuses["prepared-manifest"] is CheckStatus.FAIL
    assert "clean-worktree" not in statuses


@pytest.mark.parametrize("change", ("unknown-source", "synthetic"))
def test_contract_rejects_unapproved_source_policy(
    tmp_path: Path,
    change: str,
) -> None:
    def mutate(recipe: dict[str, Any]) -> None:
        sources = recipe["sources"]
        assert isinstance(sources, list)
        if change == "unknown-source":
            extra = dict(sources[-1])
            extra["source_id"] = "unapproved"
            extra["selection"] = {"strategy": "stable-hash-rank"}
            sources.append(extra)
        else:
            oasst = sources[0]
            assert isinstance(oasst, dict)
            selection = oasst["selection"]
            assert isinstance(selection, dict)
            selection["reject_synthetic"] = False

    spec = _write_fixture(tmp_path, mutate_recipe=mutate)
    report = verify_sft_data_gate(spec, workdir=tmp_path)

    assert report.exit_code == 1
    assert _statuses(report)["public-source-policy"] is CheckStatus.FAIL


def test_preflight_rejects_dirty_worktree_after_inputs_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _write_fixture(tmp_path)
    monkeypatch.setattr(
        sft_data_gate,
        "_base_initialization_check",
        lambda _spec, _root: _pass("base-initialization"),
    )
    monkeypatch.setattr(
        sft_data_gate,
        "_load_canonical_source",
        lambda _spec: (({"id": "one"},), _pass("canonical-source")),
    )
    monkeypatch.setattr(
        sft_data_gate,
        "_load_prepared_manifest",
        lambda _spec: (object(), _pass("prepared-manifest")),
    )
    monkeypatch.setattr(
        sft_data_gate,
        "_load_tokenizer",
        lambda _spec: (object(), _pass("tokenizer-identity")),
    )
    monkeypatch.setattr(
        sft_data_gate,
        "_split_artifact_check",
        lambda _spec: _pass("split-artifacts"),
    )
    monkeypatch.setattr(
        sft_data_gate,
        "_prepared_content_checks",
        lambda *_args: [_pass("prepared-content")],
    )

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments[0] == "rev-parse":
            return "1" * 40
        return " M configs/data/sft-public-balanced-v1.yaml"

    monkeypatch.setattr(sft_data_gate, "_run_git", fake_git)
    report = verify_sft_data_gate(
        spec,
        workdir=tmp_path,
        check_inputs=True,
        check_git=True,
    )

    assert report.exit_code == 1
    assert report.scope == "preflight"
    assert _statuses(report)["clean-worktree"] is CheckStatus.FAIL


def test_spec_rejects_unknown_fields(tmp_path: Path) -> None:
    def mutate(spec: dict[str, Any]) -> None:
        spec["evaluation_gate"] = "not-part-of-stage-1"

    spec = _write_fixture(tmp_path, mutate_spec=mutate)

    with pytest.raises(ConfigError, match="unknown evaluation_gate"):
        verify_sft_data_gate(spec, workdir=tmp_path)


def test_materialized_fixture_passes_all_input_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = NativeTokenizer.from_directory(PROJECT_ROOT / "models/base-v1-final")
    source_path = tmp_path / "data/raw/sft-public-balanced-v1/source.jsonl"
    source_path.parent.mkdir(parents=True)
    source_uri = (
        "hf://datasets/OpenAssistant/oasst1@"
        "fdf72ae0827c1cda404aff25b6603abec9e3399b/"
        "data/train-00000-of-00001-b42a775f407cee45.parquet"
    )
    requested = [
        *(
            ("train", language, task)
            for language in ("en", "zh")
            for task in (
                "general",
                "short_qa",
                "classification",
                "numeric",
                "structured",
            )
        ),
        ("dev", "en", "general"),
        ("dev", "zh", "general"),
        ("test", "en", "general"),
        ("test", "zh", "general"),
    ]
    records: list[dict[str, Any]] = []
    candidate = 0
    for expected_split, language, task in requested:
        while True:
            source_id = f"oasst1:fixture-{candidate}"
            candidate += 1
            record = {
                "id": f"record-{candidate}",
                "source_id": source_id,
                "language": language,
                "messages": [
                    {"role": "user", "content": f"Question {candidate}"},
                    {"role": "assistant", "content": f"Answer {candidate}"},
                ],
                "metadata": {
                    "source": source_uri,
                    "transform": "oasst-ranked-conversation-v1",
                    "task": task,
                },
            }
            assigned = split_records(
                [record],
                ratios=SplitRatios(),
                seed=42,
                group_by="source_id",
            )
            if assigned[expected_split]:
                records.append(record)
                break
    source_path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    prepared_path = tmp_path / "data/prepared/sft-public-balanced-v1"
    prepare_dataset(
        source_path,
        prepared_path,
        dataset_id="sft-public-balanced-v1",
        record_kind=RecordKind.SFT,
        license_name="Apache-2.0 AND CC-BY-SA-3.0 AND CC-BY-SA-4.0",
        seed=42,
        group_by="source_id",
    )
    expected = split_records(
        records,
        ratios=SplitRatios(),
        seed=42,
        group_by="source_id",
    )
    train = expected["train"]
    supervised_tokens = sum(
        sft_data_gate._supervised_token_count(
            record["messages"],
            tokenizer=tokenizer,
            max_sequence_length=512,
        )
        for record in train
    )
    source_hash = _sha256(source_path)

    def mutate_recipe(recipe: dict[str, Any]) -> None:
        output = recipe["output"]
        assert isinstance(output, dict)
        output["records"] = len(records)
        output["sha256"] = source_hash

    def mutate_spec(spec: dict[str, Any]) -> None:
        identity = spec["identity"]
        quality = spec["quality"]
        assert isinstance(identity, dict)
        assert isinstance(quality, dict)
        identity["source_records"] = len(records)
        identity["source_sha256"] = source_hash
        quality["train_examples"] = len(train)
        quality["train_supervised_tokens"] = supervised_tokens
        quality["train_language_examples"] = {"en": 5, "zh": 5}
        quality["train_task_examples"] = {
            "general": 2,
            "short_qa": 2,
            "classification": 2,
            "numeric": 2,
            "structured": 2,
        }

    spec = _write_fixture(
        tmp_path,
        mutate_recipe=mutate_recipe,
        mutate_spec=mutate_spec,
    )
    monkeypatch.setattr(
        sft_data_gate,
        "_base_initialization_check",
        lambda _spec, _root: _pass("base-initialization"),
    )
    monkeypatch.setattr(
        sft_data_gate,
        "_load_tokenizer",
        lambda _spec: (tokenizer, _pass("tokenizer-identity")),
    )

    report = verify_sft_data_gate(
        spec,
        workdir=tmp_path,
        check_inputs=True,
    )

    assert report.exit_code == 0
    assert report.counts == {"pass": 14, "warn": 0, "fail": 0}


def test_supervised_token_count_is_assistant_only_and_strict() -> None:
    tokenizer = NativeTokenizer.from_directory(PROJECT_ROOT / "models/base-v1-final")
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]

    count = sft_data_gate._supervised_token_count(
        messages,
        tokenizer=tokenizer,
        max_sequence_length=512,
    )

    assert count > 0
    with pytest.raises(ContractError, match="must end with assistant"):
        sft_data_gate._supervised_token_count(
            messages[:1],
            tokenizer=tokenizer,
            max_sequence_length=512,
        )
