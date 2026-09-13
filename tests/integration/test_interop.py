from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_lifecycle_lab.chat import ChatSession
from llm_lifecycle_lab.contracts import (
    CheckpointMetadata,
    ModelRoute,
    RunConfig,
    RunProfile,
    Stage,
)
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.prepare import prepare_dataset
from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError, ContractError
from llm_lifecycle_lab.interop.assets import (
    TRANSFORMERS_VERSION,
    verify_hf_assets,
    write_hf_manifest,
)
from llm_lifecycle_lab.interop.benchmark import benchmark_hf_cpu
from llm_lifecycle_lab.interop.evaluation import HFEvaluator
from llm_lifecycle_lab.interop.export import export_native_hf, export_tokenizer
from llm_lifecycle_lab.interop.transfer import QWEN_ID, load_transfer_bundle
from llm_lifecycle_lab.model.native import NativeModelConfig, NativeTransformer
from llm_lifecycle_lab.model.protocol import GenerationConfig
from llm_lifecycle_lab.tokenizer import NativeTokenizer, train_native_tokenizer
from llm_lifecycle_lab.training.transfer import run_transfer_sft

pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("psutil")

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "configs/evaluation/lifecycle-v1.yaml"
REVISION = "a" * 40


def make_native_checkpoint(
    tmp_path: Path, *, qk_norm: bool = False
) -> tuple[Path, Path]:
    rows = [
        {
            "id": f"{language}-{index}",
            "source_id": f"group-{index}",
            "source": "self-authored",
            "language": language,
            "text": text * 4,
        }
        for index in range(80)
        for language, text in (
            ("en", f"Number {index}. A short English training sentence. "),
            ("zh", f"编号{index}。这是一句简短的中文训练文本。"),
        )
    ]
    source = tmp_path / "pretrain.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    prepared = tmp_path / "prepared"
    prepare_dataset(
        source,
        prepared,
        dataset_id="interop-fixture",
        record_kind="pretrain",
        license_name="Apache-2.0",
        group_by="source_id",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = train_native_tokenizer(
        prepared / "data_manifest.json",
        tokenizer_dir,
        tokenizer_id="interop-fixture",
        vocab_size=320,
    )
    config = NativeModelConfig(
        model_id="interop-fixture",
        vocab_size=tokenizer.vocab_size,
        num_hidden_layers=1,
        hidden_size=24,
        num_attention_heads=3,
        num_key_value_heads=1,
        intermediate_size=48,
        max_sequence_length=512,
        qk_norm=qk_norm,
    )
    checkpoint = tmp_path / "checkpoint"
    NativeTransformer(config).save(checkpoint / "model")
    metadata = CheckpointMetadata(
        checkpoint_id="step-00000000",
        run_id="interop-fixture",
        model_route=ModelRoute.NATIVE,
        stage=Stage.PRETRAIN,
        step=0,
        tokenizer_sha256=tokenizer.content_sha256,
        config_sha256="a" * 64,
    )
    (checkpoint / "checkpoint_metadata.json").write_text(
        metadata.to_json(), encoding="utf-8"
    )
    return checkpoint, tokenizer_dir


def make_qwen_snapshot(
    tmp_path: Path, tokenizer_dir: Path
) -> tuple[Path, NativeTokenizer]:
    import transformers

    tokenizer = NativeTokenizer.from_directory(tokenizer_dir)
    snapshot = tmp_path / "qwen-snapshot"
    snapshot.mkdir()
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=tokenizer.vocab_size,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=512,
            tie_word_embeddings=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    model.save_pretrained(snapshot, safe_serialization=True)
    export_tokenizer(tokenizer_dir, tokenizer, snapshot, 512)
    write_hf_manifest(
        snapshot,
        {
            "kind": "qwen-snapshot",
            "model_route": "qwen3-transfer",
            "stage": "pretrain",
            "repo_id": QWEN_ID,
            "revision": REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "weights_origin": "hub",
        },
    )
    return snapshot, tokenizer


def make_sft_manifest(tmp_path: Path) -> Path:
    rows = [
        {
            "id": f"{language}-{index}",
            "source_id": f"example-{index}",
            "language": language,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": str(index)},
            ],
        }
        for index in range(80)
        for language, prompt in (
            ("en", f"Write the integer {index} using digits only."),
            ("zh", f"请只用数字写出整数{index}。"),
        )
    ]
    source = tmp_path / "sft.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    prepared = tmp_path / "sft-data"
    prepare_dataset(
        source,
        prepared,
        dataset_id="interop-sft-fixture",
        record_kind="sft",
        license_name="Apache-2.0",
        group_by="source_id",
    )
    return prepared / "data_manifest.json"


def transfer_settings(snapshot: Path) -> dict:
    return {
        "provider": "huggingface",
        "model_id": QWEN_ID,
        "revision": REVISION,
        "tokenizer_revision": REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "snapshot": str(snapshot),
        "training_method": "lora",
        "lora": {
            "rank": 2,
            "alpha": 4,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
        },
    }


@pytest.mark.parametrize(
    ("qk_norm", "architecture"), [(False, "llama"), (True, "qwen3")]
)
def test_native_hf_export_is_verified_and_tamper_evident(
    tmp_path: Path, qk_norm: bool, architecture: str
) -> None:
    checkpoint, tokenizer = make_native_checkpoint(tmp_path, qk_norm=qk_norm)
    output = tmp_path / "hf-export"
    manifest = export_native_hf(checkpoint, output, tokenizer_dir=tokenizer)
    assert manifest["architecture"] == architecture
    verification = json.loads((output / "verification.json").read_text())
    assert verification["cache_matches"]
    assert verification["chat_tokens_match"]
    assert verification["greedy_matches"]
    evaluator = HFEvaluator(output)
    result = evaluator.generate(
        evaluator.tokenizer.encode("Once", add_bos=True), max_new_tokens=2
    )
    assert len(result["token_ids"]) <= 2
    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        export_native_hf(checkpoint, output, tokenizer_dir=tokenizer)
    (output / "config.json").write_text("{}")
    with pytest.raises(ArtifactError, match="content/hash mismatch"):
        verify_hf_assets(output)


def test_hf_cpu_benchmark_records_fp32_and_dynamic_int8(tmp_path: Path) -> None:
    checkpoint, tokenizer = make_native_checkpoint(tmp_path)
    output = tmp_path / "hf-export"
    export_native_hf(checkpoint, output, tokenizer_dir=tokenizer)
    supported = set(torch.backends.quantized.supported_engines)
    dynamic_int8 = bool(supported & {"x86", "fbgemm", "onednn", "qnnpack"})
    report = benchmark_hf_cpu(
        output,
        max_new_tokens=1,
        repeats=1,
        threads=1,
        dynamic_int8=dynamic_int8,
    )
    assert report["protocol"]["runtime"]["transformers_version"] == (
        TRANSFORMERS_VERSION
    )
    assert len(report["modes"]["fp32"]["samples"]) == 4
    assert report["modes"]["fp32"]["state_dict_bytes"] > 0
    if dynamic_int8:
        quantized = report["modes"]["dynamic-int8"]
        assert report["protocol"]["quantized_engine"] in supported
        assert len(quantized["samples"]) == 4
        assert 0 <= quantized["greedy_sequence_match_fraction"] <= 1
        assert quantized["state_size_ratio_to_fp32"] > 0
    else:
        assert report["protocol"]["quantized_engine"] is None
        assert set(report["modes"]) == {"fp32"}


def test_qwen_lora_route_freezes_base_and_restores_checkpoint(tmp_path: Path) -> None:
    _, tokenizer_dir = make_native_checkpoint(tmp_path)
    snapshot, _ = make_qwen_snapshot(tmp_path, tokenizer_dir)
    manifest = make_sft_manifest(tmp_path)
    settings = transfer_settings(snapshot)
    base_hash = sha256_file(snapshot / "model.safetensors")
    bundle, binding = load_transfer_bundle(snapshot, settings)
    assert binding["revision"] == REVISION
    assert all("lora_" in name for name, _ in bundle.model.trainable_parameters())
    config = RunConfig(
        model_route=ModelRoute.QWEN3_TRANSFER,
        run_profile=RunProfile.LEARN,
        stage=Stage.SFT,
        model=settings,
        data={"manifest": str(manifest), "evaluation_suite": str(SUITE)},
        training={
            "sequence_length": 128,
            "max_steps": 2,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.001,
            "device": "cpu",
            "checkpoint_interval": 1,
            "eval_interval": 1,
            "eval_batches": 2,
        },
        output_dir=str(tmp_path / "runs"),
    )
    run = run_transfer_sft(config, run_id="qwen-lora", workdir=ROOT)
    assert run.result.global_step == 2
    assert sha256_file(snapshot / "model.safetensors") == base_hash
    final = Path(run.result.final_checkpoint)
    assert (final / "model/adapter_model.safetensors").is_file()
    assert not (final / "model/model.safetensors").exists()
    evaluator = HFEvaluator(final, base_model=snapshot)
    assert evaluator.identity["model_route"] == "qwen3-transfer"
    assert len(evaluator.generate([1], max_new_tokens=1)["token_ids"]) == 1
    with pytest.raises(ConfigError, match="reached or exceeded"):
        run_transfer_sft(config, resume_run="qwen-lora", workdir=ROOT)


def test_qwen_route_rejects_revision_adapter_and_snapshot_mismatch(
    tmp_path: Path,
) -> None:
    _, tokenizer_dir = make_native_checkpoint(tmp_path)
    snapshot, _ = make_qwen_snapshot(tmp_path, tokenizer_dir)
    settings = transfer_settings(snapshot)
    with pytest.raises(ContractError, match="revisions disagree"):
        load_transfer_bundle(snapshot, {**settings, "tokenizer_revision": "b" * 40})
    with pytest.raises(ContractError, match="invalid LoRA"):
        load_transfer_bundle(snapshot, {**settings, "lora": {"rank": 0}})
    (snapshot / "config.json").write_text("{}")
    with pytest.raises(ArtifactError, match="content/hash mismatch"):
        load_transfer_bundle(snapshot, settings)


def test_chat_defaults_base_to_completion_and_bounds_history(tmp_path: Path) -> None:
    checkpoint, tokenizer = make_native_checkpoint(tmp_path)
    export = tmp_path / "hf-export"
    export_native_hf(checkpoint, export, tokenizer_dir=tokenizer)
    evaluator = HFEvaluator(export)
    completion = ChatSession(
        evaluator,
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
    )
    assert completion.mode == "completion"
    assert completion.respond("Once")["generated_tokens"] == 2
    with pytest.raises(ContractError, match="system prompt"):
        ChatSession(
            evaluator,
            system="system",
            mode="completion",
            generation=GenerationConfig(max_new_tokens=2),
        )
    chat = ChatSession(
        evaluator,
        mode="chat",
        history_policy="drop-oldest",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
    )
    first = chat.respond("hello")
    second = chat.respond("again")
    assert first["dropped_turns"] == second["dropped_turns"] == 0
    chat.reset()
    assert chat.history == []
