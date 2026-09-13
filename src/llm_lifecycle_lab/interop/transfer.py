"""Pinned Qwen snapshots and local-only loading; never fall back to random weights."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab.artifacts import RunArtifacts
from llm_lifecycle_lab.config import canonical_json, dump_yaml
from llm_lifecycle_lab.contracts import ModelMetadata, ModelRoute, utc_now
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.exceptions import ArtifactError, ContractError
from llm_lifecycle_lab.interop.assets import (
    TRANSFORMERS_VERSION,
    require_hf,
    verify_hf_assets,
    write_hf_manifest,
)
from llm_lifecycle_lab.interop.backend import HFModelAdapter, HFTokenizer, apply_lora
from llm_lifecycle_lab.model.bundle import ModelBundle
from llm_lifecycle_lab.tokenizer.native import NATIVE_CHAT_TEMPLATE_VERSION

QWEN_ID = "Qwen/Qwen3-0.6B-Base"


def prepare_qwen_snapshot(
    output: str | Path, *, revision: str, offline: bool = False
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ContractError("Qwen revision must be a pinned 40-character commit SHA")
    hf = require_hf()
    from huggingface_hub import snapshot_download

    target = Path(output).resolve()
    if target.exists():
        raise ArtifactError("Qwen snapshot output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))
    try:
        snapshot_download(
            QWEN_ID,
            revision=revision,
            local_dir=temporary,
            local_files_only=offline,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "merges.txt",
                "vocab.json",
                "*.safetensors",
                "*.safetensors.index.json",
                "LICENSE*",
                "README.md",
            ],
        )
        shutil.rmtree(temporary / ".cache", ignore_errors=True)
        config = hf.AutoConfig.from_pretrained(
            temporary, local_files_only=True, trust_remote_code=False
        )
        if config.model_type != "qwen3" or not list(temporary.glob("*.safetensors")):
            raise ArtifactError("Qwen snapshot has no compatible config/weights")
        pipeline = {
            "schema_version": "1.0",
            "model_route": "qwen3-transfer",
            "run_profile": "learn",
            "stage": "sft",
            "seed": 42,
            "output_dir": "runs",
            "model": {
                "provider": "huggingface",
                "model_id": QWEN_ID,
                "revision": revision,
                "tokenizer_revision": revision,
                "transformers_version": TRANSFORMERS_VERSION,
                "snapshot": str(target),
                "training_method": "lora",
                "lora": {
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.0,
                    "target_modules": ["q_proj", "v_proj"],
                },
            },
            "data": {
                "manifest": "data/prepared/sft-bilingual-v1/data_manifest.json",
                "evaluation_suite": "configs/evaluation/lifecycle-v1.yaml",
            },
            "training": {
                "device": "cpu",
                "dtype": "float32",
                "sequence_length": 512,
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.0001,
                "checkpoint_interval": 1,
                "eval_interval": 1,
                "eval_batches": 2,
            },
        }
        RunArtifacts(target.name, temporary).write_text(
            "sft.example.yaml", dump_yaml(pipeline)
        )
        write_hf_manifest(
            temporary,
            {
                "kind": "qwen-snapshot",
                "model_route": "qwen3-transfer",
                "stage": "pretrain",
                "repo_id": QWEN_ID,
                "revision": revision,
                "transformers_version": TRANSFORMERS_VERSION,
                "weights_origin": "hub",
                "created_at": utc_now(),
            },
        )
        result = verify_hf_assets(temporary)
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def load_transfer_bundle(
    snapshot: str | Path, settings: dict[str, Any]
) -> tuple[ModelBundle, dict[str, Any]]:
    method = settings.get("training_method", "lora")
    if method not in {"lora", "full"}:
        raise ContractError(
            "training_method must be lora or full; QLoRA is not enabled"
        )
    hf = require_hf(peft=method == "lora")
    directory = Path(snapshot)
    manifest = verify_hf_assets(directory)
    expected = {
        "kind": "qwen-snapshot",
        "repo_id": QWEN_ID,
        "model_route": "qwen3-transfer",
        "revision": settings.get("revision"),
        "weights_origin": "hub",
        "transformers_version": settings.get("transformers_version"),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ArtifactError("Qwen snapshot does not match the pinned model settings")
    if (
        settings.get("model_id") != QWEN_ID
        or settings.get("revision") != settings.get("tokenizer_revision")
        or settings.get("transformers_version") != TRANSFORMERS_VERSION
    ):
        raise ContractError("Qwen model, tokenizer and Transformers revisions disagree")
    model = hf.AutoModelForCausalLM.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    if model.config.model_type != "qwen3":
        raise ContractError("Transfer route only supports Qwen3")
    tokenizer = HFTokenizer(
        hf.AutoTokenizer.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
        ),
        content_sha256=sha256_file(directory / "tokenizer.json"),
    )
    if tokenizer.vocab_size > model.config.vocab_size:
        raise ContractError("Qwen tokenizer IDs exceed the model vocabulary")
    lora = json.loads(canonical_json(dict(settings.get("lora", {}))))
    binding = {
        "snapshot_manifest_sha256": sha256_file(directory / "hf_manifest.json"),
        "repo_id": QWEN_ID,
        "revision": manifest["revision"],
        "transformers_version": TRANSFORMERS_VERSION,
        "training_method": method,
        "lora": lora if method == "lora" else {},
    }
    if method == "lora":
        model = apply_lora(model, lora)
    adapter = HFModelAdapter(model, binding=binding)
    metadata = ModelMetadata(
        model_route=ModelRoute.QWEN3_TRANSFER,
        provider="huggingface",
        model_id=QWEN_ID,
        architecture="qwen3",
        upstream_revision=manifest["revision"],
        tokenizer_revision=manifest["revision"],
        chat_template_version=NATIVE_CHAT_TEMPLATE_VERSION,
        parameter_count=sum(p.numel() for p in model.parameters()),
        capabilities=("forward", "generate", method),
    )
    return ModelBundle(adapter, tokenizer, tokenizer.chat_template, metadata), binding
