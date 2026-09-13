from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from llm_lifecycle_lab.contracts import CheckpointMetadata, ModelRoute, Stage
from llm_lifecycle_lab.data.prepare import prepare_dataset
from llm_lifecycle_lab.evaluation.corpus import scan_corpus
from llm_lifecycle_lab.evaluation.native import NativeEvaluator
from llm_lifecycle_lab.evaluation.report import compare_reports
from llm_lifecycle_lab.evaluation.runner import evaluate_capabilities
from llm_lifecycle_lab.evaluation.suite import (
    digest,
    load_suite,
    repeated_ngram_fraction,
    score_response,
)
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.native import NativeModelConfig, NativeTransformer
from llm_lifecycle_lab.tokenizer import NativeTokenizer, train_native_tokenizer

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "configs/evaluation/lifecycle-v1.yaml"


@pytest.fixture
def native_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    rows = [
        {
            "id": f"{language}-{i}",
            "source_id": f"group-{i}",
            "source": "self-authored",
            "language": language,
            "text": text * 4,
        }
        for i in range(80)
        for language, text in (
            ("en", f"Reading number {i}. A quiet library is open every morning. "),
            ("zh", f"阅读编号{i}。安静的图书馆每天上午开放。"),
        )
    ]
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    prepare_dataset(
        source,
        tmp_path / "prepared",
        dataset_id="eval-fixture",
        record_kind="pretrain",
        license_name="Apache-2.0",
        group_by="source_id",
    )
    manifest = tmp_path / "prepared/data_manifest.json"
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = train_native_tokenizer(
        manifest, tokenizer_dir, tokenizer_id="eval-fixture", vocab_size=320
    )
    config = NativeModelConfig(
        model_id="eval-fixture",
        vocab_size=tokenizer.vocab_size,
        num_hidden_layers=1,
        hidden_size=24,
        num_attention_heads=3,
        num_key_value_heads=1,
        intermediate_size=48,
        max_sequence_length=1024,
    )
    checkpoint = tmp_path / "checkpoint"
    NativeTransformer(config).save(checkpoint / "model")
    metadata = CheckpointMetadata(
        checkpoint_id="step-00000000",
        run_id="eval-fixture",
        model_route=ModelRoute.NATIVE,
        stage=Stage.PRETRAIN,
        step=0,
        tokenizer_sha256=tokenizer.content_sha256,
        config_sha256="a" * 64,
    )
    (checkpoint / "checkpoint_metadata.json").write_text(metadata.to_json())
    return checkpoint, tokenizer_dir, manifest


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"count":4}', 1),
        (' {"count":4} \n', 1),
        ('{"count":true}', 0),
        ('{"count":4.0}', 0),
        ('{"count":4,"extra":1}', 0),
        ('```{"count":4}```', 0),
        ('{"count":0,"count":4}', 0),
        ('{"count":NaN}', 0),
        ('{"count":1e999}', 0),
        ("[]", 0),
    ],
)
def test_strict_json_scoring(text: str, expected: float) -> None:
    assert score_response(text, {"type": "json", "answer": {"count": 4}}) == expected


def test_exact_scoring_and_repetition() -> None:
    assert score_response(" yes\n", {"type": "exact", "answer": "yes"}) == 1
    assert score_response("yes, correct", {"type": "exact", "answer": "yes"}) == 0
    assert repeated_ngram_fraction([1, 2]) == 0
    assert repeated_ngram_fraction([1, 1, 1, 1, 1]) == pytest.approx(2 / 3)


def test_uniform_likelihood_is_length_normalized(native_fixture) -> None:
    import math

    checkpoint, tokenizer, _ = native_fixture
    evaluator = NativeEvaluator(checkpoint, tokenizer_dir=tokenizer)
    with torch.no_grad():
        for parameter in evaluator.model.parameters():
            parameter.zero_()
    short = evaluator.nll([1], [2])
    long = evaluator.nll([1], [2, 3, 4]) / 3
    assert short == pytest.approx(math.log(evaluator.tokenizer.vocab_size), abs=1e-6)
    assert short == long


def test_multiturn_uses_generated_history(
    native_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []

    def generate(self, ids, *, max_new_tokens, protocol="raw"):
        seen.append(self.tokenizer.decode(ids, skip_special_tokens=False))
        return {
            "text": "actual-model-reply",
            "token_ids": [17],
            "stop_reason": "length",
        }

    monkeypatch.setattr(NativeEvaluator, "generate", generate)
    checkpoint, tokenizer, _ = native_fixture
    evaluate_capabilities(
        checkpoint=checkpoint,
        tokenizer_dir=tokenizer,
        suite_path=SUITE,
        output=tmp_path / "history",
        max_new_tokens=4,
    )
    second = next(text for text in seen if "What was the code word?" in text)
    assert "Assistant: actual-model-reply" in second
    assert "Assistant: OK" not in second


def test_suite_rejects_duplicate_ids_and_invalid_rules(tmp_path: Path) -> None:
    suite = load_suite(SUITE)
    suite["cases"].append(suite["cases"][0])
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ContractError, match="duplicate"):
        load_suite(path)
    with pytest.raises(ContractError, match="rule.type"):
        score_response("x", {"type": "python-eval", "answer": "x"})


def test_complete_evaluation_and_protocol_gate(native_fixture, tmp_path: Path) -> None:
    checkpoint, tokenizer, manifest = native_fixture
    first = evaluate_capabilities(
        checkpoint=checkpoint,
        tokenizer_dir=tokenizer,
        suite_path=SUITE,
        output=tmp_path / "first",
        max_new_tokens=4,
    )
    assert first["metrics"]["overlap.query_fraction"]["status"] == "not-checked"
    assert first["metrics"]["memory.train.suffix_exact"]["value"] is None
    assert first["metrics"]["corpus.bpb.en"]["count"] > 0
    assert first["metrics"]["corpus.bpb.zh"]["count"] > 0
    assert all("baseline" in value for value in first["metrics"].values())
    assert first["metrics"]["preference.accuracy"]["baseline"]["value"] == 0.5
    assert first["metrics"]["multiturn.success"]["count"] == 6
    assert (tmp_path / "first/report.md").is_file()
    second = evaluate_capabilities(
        checkpoint=checkpoint,
        tokenizer_dir=tokenizer,
        suite_path=SUITE,
        output=tmp_path / "second",
        max_new_tokens=4,
        baseline_path=tmp_path / "first/report.json",
    )
    assert first["metrics"] == second["metrics"]
    for row in second["comparison"]["metrics"].values():
        assert row["delta_from_first"][-1] in (0, None)
    with pytest.raises(ContractError, match="different protocol"):
        evaluate_capabilities(
            checkpoint=checkpoint,
            tokenizer_dir=tokenizer,
            suite_path=SUITE,
            output=tmp_path / "incomparable",
            max_new_tokens=5,
            baseline_path=tmp_path / "first/report.json",
        )
    with pytest.raises(ContractError, match="output exists"):
        evaluate_capabilities(
            checkpoint=checkpoint,
            tokenizer_dir=tokenizer,
            suite_path=SUITE,
            output=tmp_path / "first",
        )
    changed = deepcopy(first)
    changed["protocol"]["tokenizer_sha256"] = "b" * 64
    changed["protocol_sha256"] = digest(changed["protocol"])
    with pytest.raises(ContractError, match="incomparable"):
        compare_reports([first, changed])


def test_corpus_audit_and_memory_probes(native_fixture, tmp_path: Path) -> None:
    checkpoint, tokenizer_dir, manifest = native_fixture
    suite = load_suite(SUITE)
    tokenizer = NativeTokenizer.from_directory(tokenizer_dir)
    train_rows = [
        json.loads(line)
        for line in (manifest.parent / "train.jsonl").read_text().splitlines()
    ]
    suite["cases"][0]["text"] = train_rows[0]["text"]
    audit = scan_corpus(manifest, suite=suite, tokenizer=tokenizer)
    assert "corpus-en-1" in audit["matched_query_ids"]
    assert audit["scanned_records"]["train"] == len(train_rows)
    assert {p["split"] for p in audit["probes"]} == {"train", "test"}
    report = evaluate_capabilities(
        checkpoint=checkpoint,
        tokenizer_dir=tokenizer_dir,
        suite_path=SUITE,
        output=tmp_path / "with-corpus",
        pretrain_manifest=manifest,
        max_new_tokens=2,
    )
    assert report["metrics"]["memory.train.suffix_exact"]["count"] == 4
    assert report["metrics"]["memory.test.suffix_exact"]["count"] == 4
    assert report["corpus_audit"]["status"] == "checked"
    changed_manifest = json.loads(manifest.read_text())
    changed_manifest["dataset_id"] = "mismatch"
    wrong = manifest.parent / "wrong.json"
    wrong.write_text(json.dumps(changed_manifest))
    with pytest.raises(ContractError, match="not bound"):
        scan_corpus(wrong, suite=suite, tokenizer=tokenizer)


def test_model_adapter_fails_on_tokenizer_or_context_mismatch(
    native_fixture, tmp_path: Path
) -> None:
    checkpoint, tokenizer, _ = native_fixture
    evaluator = NativeEvaluator(checkpoint, tokenizer_dir=tokenizer)
    with pytest.raises(ContractError, match="overflow"):
        evaluator.generate([1] * 1024, max_new_tokens=1)
    metadata_path = checkpoint / "checkpoint_metadata.json"
    data = json.loads(metadata_path.read_text())
    data["tokenizer_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(data))
    with pytest.raises(ContractError, match="hashes differ"):
        NativeEvaluator(checkpoint, tokenizer_dir=tokenizer)


def test_sft_masks_only_assistant_content_and_end(native_fixture) -> None:
    from llm_lifecycle_lab.data.sft import collate_sft, encode_sft_example

    _, directory, _ = native_fixture
    tokenizer = NativeTokenizer.from_directory(directory)
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "再说一次。"},
        {"role": "assistant", "content": "你好。"},
    ]
    example = encode_sft_example(
        messages,
        tokenizer=tokenizer,
        sequence_length=512,
        language="zh",
    )
    labels = example["labels"]
    supervised = labels[labels != -100].tolist()
    expected = (
        tokenizer.encode("Hello.")
        + [tokenizer.chat_end_token_id]
        + tokenizer.encode("你好。")
        + [tokenizer.chat_end_token_id]
    )
    assert supervised == expected
    assert example["input_ids"].tolist() == tokenizer.encode_chat(messages)
    short = encode_sft_example(
        messages[1:3],
        tokenizer=tokenizer,
        sequence_length=512,
        language="en",
    )
    batch = collate_sft([example, short], pad_token_id=tokenizer.pad_token_id)
    padding = ~batch["attention_mask"]
    assert padding.any()
    assert (batch["labels"][padding] == -100).all()
    with pytest.raises(ContractError, match="no implicit truncation"):
        encode_sft_example(
            messages,
            tokenizer=tokenizer,
            sequence_length=2,
            language="en",
        )
    with pytest.raises(ContractError, match="end with assistant"):
        encode_sft_example(
            messages[:-1],
            tokenizer=tokenizer,
            sequence_length=512,
            language="en",
        )
    with pytest.raises(ContractError, match="expected user"):
        encode_sft_example(
            [{"role": "assistant", "content": "No user"}],
            tokenizer=tokenizer,
            sequence_length=512,
            language="en",
        )


def test_sft_new_stage_and_frozen_resume_inputs(
    native_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from llm_lifecycle_lab.contracts import RunConfig, RunProfile
    from llm_lifecycle_lab.data.fingerprint import sha256_file
    from llm_lifecycle_lab.data.sft import load_sft_splits
    from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError
    from llm_lifecycle_lab.training.sft import run_native_sft

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from sft_experiment import prepare_sft_fixture

    checkpoint, tokenizer, _ = native_fixture
    manifest = prepare_sft_fixture(tmp_path)
    config = RunConfig(
        model_route=ModelRoute.NATIVE,
        stage=Stage.SFT,
        run_profile=RunProfile.SMOKE,
        model={
            "provider": "native",
            "model_id": "eval-fixture",
            "init_checkpoint": str(checkpoint),
            "tokenizer": str(tokenizer),
        },
        data={"manifest": str(manifest), "evaluation_suite": str(SUITE)},
        output_dir=str(tmp_path / "sft-runs"),
        training={
            "sequence_length": 512,
            "max_steps": 2,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "device": "cpu",
            "eval_batches": 2,
            "eval_interval": 1,
            "checkpoint_interval": 1,
        },
    )
    parent_hash = sha256_file(checkpoint / "model/model.pt")
    run = run_native_sft(config, run_id="sft-test", workdir=ROOT)
    assert run.result.global_step == 2
    assert run.result.tokens_seen > 0
    assert sha256_file(checkpoint / "model/model.pt") == parent_hash
    final = Path(run.result.final_checkpoint)
    assert sha256_file(final / "model/model.pt") != parent_hash
    assert (
        json.loads((final / "checkpoint_metadata.json").read_text())["stage"] == "sft"
    )
    initialization = json.loads(
        (run.artifacts.path / "initialization.json").read_text()
    )
    assert initialization["parent_weights_sha256"] == parent_hash
    assert initialization["optimizer_inherited"] is False
    metrics = [
        json.loads(line)
        for line in (run.artifacts.path / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[0]["step"] == 0
    assert metrics[0]["eval_en_tokens"] > 0 and metrics[0]["eval_zh_tokens"] > 0
    with pytest.raises(ConfigError, match="reached or exceeded"):
        run_native_sft(config, resume_run="sft-test", workdir=ROOT)
    changed = replace(config, training={**config.training, "max_steps": 3})
    with pytest.raises(ArtifactError, match="resolved config"):
        run_native_sft(changed, resume_run="sft-test", workdir=ROOT)
    changed_manifest = json.loads(manifest.read_text())
    changed_manifest["license"] = "changed"
    manifest.write_text(json.dumps(changed_manifest))
    with pytest.raises(ArtifactError, match="frozen initialization"):
        run_native_sft(config, resume_run="sft-test", workdir=ROOT)

    # Rebuild a valid manifest containing a protected evaluation prompt.
    rows = [
        json.loads(line)
        for line in (tmp_path / "sft-source.jsonl").read_text().splitlines()
    ]
    rows[0]["messages"][0]["content"] = (
        "Return only the text inside brackets: [silver]."
    )
    source = tmp_path / "leaky-sft.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prepare_dataset(
        source,
        tmp_path / "leaky",
        dataset_id="leaky",
        record_kind="sft",
        license_name="Apache-2.0",
        group_by="source_id",
    )
    with pytest.raises(ContractError, match="overlaps the fixed evaluation"):
        load_sft_splits(
            tmp_path / "leaky/data_manifest.json",
            tokenizer=NativeTokenizer.from_directory(tokenizer),
            sequence_length=512,
            evaluation_suite=SUITE,
        )
