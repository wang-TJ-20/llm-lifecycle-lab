"""Export Native weights to standard HF classes, with mandatory CPU parity checks."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from tokenizers.processors import TemplateProcessing

from llm_lifecycle_lab.artifacts import RunArtifacts
from llm_lifecycle_lab.contracts import utc_now
from llm_lifecycle_lab.evaluation.native import NativeEvaluator
from llm_lifecycle_lab.exceptions import ArtifactError
from llm_lifecycle_lab.interop.assets import (
    TRANSFORMERS_VERSION,
    require_hf,
    verify_hf_assets,
    write_hf_manifest,
)
from llm_lifecycle_lab.tokenizer.native import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
)


def to_hf_config(native, tokenizer):
    hf = require_hf()
    factory = hf.Qwen3Config if native.qk_norm else hf.LlamaConfig
    return factory(
        vocab_size=native.vocab_size,
        hidden_size=native.hidden_size,
        intermediate_size=native.intermediate_size,
        num_hidden_layers=native.num_hidden_layers,
        num_attention_heads=native.num_attention_heads,
        num_key_value_heads=native.num_key_value_heads,
        head_dim=native.head_dim,
        max_position_embeddings=native.max_sequence_length,
        rms_norm_eps=native.norm_eps,
        rope_theta=native.rope_theta,
        attention_dropout=native.attention_dropout,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        tie_word_embeddings=native.tie_word_embeddings,
        initializer_range=native.initializer_range,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def mapped_state(native) -> dict[str, torch.Tensor]:
    mapping = {
        "token_embedding.weight": "model.embed_tokens.weight",
        "final_norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for layer in range(native.config.num_hidden_layers):
        for source, target in (
            ("input_norm", "input_layernorm"),
            ("post_attention_norm", "post_attention_layernorm"),
            ("attention.q_proj", "self_attn.q_proj"),
            ("attention.k_proj", "self_attn.k_proj"),
            ("attention.v_proj", "self_attn.v_proj"),
            ("attention.o_proj", "self_attn.o_proj"),
            ("mlp.gate_proj", "mlp.gate_proj"),
            ("mlp.up_proj", "mlp.up_proj"),
            ("mlp.down_proj", "mlp.down_proj"),
        ):
            mapping[f"layers.{layer}.{source}.weight"] = (
                f"model.layers.{layer}.{target}.weight"
            )
        if native.config.qk_norm:
            for projection in ("q", "k"):
                mapping[f"layers.{layer}.attention.{projection}_norm.weight"] = (
                    f"model.layers.{layer}.self_attn.{projection}_norm.weight"
                )
    state = native.state_dict()
    if state.keys() != mapping.keys():
        raise ArtifactError("unmapped Native weights; export cannot drop parameters")
    # Both implementations use split-half RoPE. No Q/K permutation is needed.
    return {
        mapping[name]: value.detach().cpu().contiguous()
        for name, value in state.items()
    }


def export_tokenizer(tokenizer_dir: Path, tokenizer, output: Path, max_length: int):
    hf = require_hf()
    fast = hf.PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        additional_special_tokens=[
            token
            for token in SPECIAL_TOKENS
            if token not in {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN}
        ],
        model_max_length=max_length,
        chat_template=tokenizer.chat_template,
    )
    # Standard tokenizer(text) adds BOS; apply_chat_template bypasses this
    # postprocessor, preserving the exact no-BOS Native conversation wire format.
    fast.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A",
        special_tokens=[(BOS_TOKEN, tokenizer.bos_token_id)],
    )
    fast.save_pretrained(output)
    return fast


@torch.inference_mode()
def verify_parity(evaluator: NativeEvaluator, hf_model, hf_tokenizer) -> dict[str, Any]:
    model, tokenizer = evaluator.model, evaluator.tokenizer
    model.eval()
    hf_model.eval()
    max_error = 0.0
    samples = []
    for text in ("Once upon a time", "从前", "北京是", "ＡＢＣ １２３"):
        ids = tokenizer.encode(text, add_bos=True)
        if ids != hf_tokenizer.encode(text):
            raise ArtifactError("exported tokenizer IDs differ from Native")
        tensor = torch.tensor([ids])
        expected = model.forward(input_ids=tensor).logits
        actual = hf_model(input_ids=tensor, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
        max_error = max(max_error, float((actual - expected).abs().max()))
        count = min(8, model.config.max_sequence_length - len(ids))
        if count <= 0:
            raise ArtifactError("model context is too short for export verification")
        native_tokens = evaluator.generate(ids, max_new_tokens=count)
        generated = hf_model.generate(
            input_ids=tensor,
            do_sample=False,
            max_new_tokens=count,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )[0, len(ids) :].tolist()
        if generated and generated[-1] == tokenizer.eos_token_id:
            generated = generated[:-1]
        if generated != native_tokens["token_ids"]:
            raise ArtifactError("exported greedy generation differs from Native")
        samples.append({"prompt": text, "generated_token_ids": generated})
    chat = [{"role": "user", "content": "你好"}]
    if tokenizer.encode_chat(chat, add_generation_prompt=True) != (
        hf_tokenizer.apply_chat_template(chat, add_generation_prompt=True)
    ):
        raise ArtifactError("exported chat template changed token IDs")
    prefix = torch.tensor([[tokenizer.bos_token_id, 17, 18, 19]])
    tail = torch.tensor([[20]])
    cache = hf_model(input_ids=prefix, use_cache=True).past_key_values
    cached = hf_model(input_ids=tail, past_key_values=cache, use_cache=True).logits
    full = model.forward(input_ids=torch.cat((prefix, tail), dim=1)).logits[:, -1:]
    torch.testing.assert_close(cached, full, rtol=1e-4, atol=2e-5)
    return {
        "device": "cpu",
        "dtype": "float32",
        "rtol": 1e-4,
        "atol": 2e-5,
        "max_logit_error": max_error,
        "greedy_matches": True,
        "chat_tokens_match": True,
        "cache_matches": True,
        "samples": samples,
    }


def export_native_hf(
    checkpoint: str | Path,
    output: str | Path,
    *,
    tokenizer_dir: str | Path | None = None,
) -> dict[str, Any]:
    hf = require_hf()
    target = Path(output)
    if target.exists():
        raise ArtifactError(f"export directory exists; refusing to overwrite: {target}")
    source = Path(checkpoint)
    tokenizer_source = Path(tokenizer_dir) if tokenizer_dir else source / "tokenizer"
    evaluator = NativeEvaluator(source, tokenizer_dir=tokenizer_source, device="cpu")
    hf_config = to_hf_config(evaluator.config, evaluator.tokenizer)
    hf_model = hf.AutoModelForCausalLM.from_config(
        hf_config, attn_implementation="sdpa"
    )
    hf_model.load_state_dict(mapped_state(evaluator.model), strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))
    try:
        hf_model.save_pretrained(temporary, safe_serialization=True)
        export_tokenizer(
            tokenizer_source,
            evaluator.tokenizer,
            temporary,
            evaluator.config.max_sequence_length,
        )
        del hf_model
        # Verify the files readers will load, not just the in-memory conversion.
        reloaded = hf.AutoModelForCausalLM.from_pretrained(
            temporary,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            torch_dtype=torch.float32,
        )
        fast = hf.AutoTokenizer.from_pretrained(
            temporary,
            local_files_only=True,
            trust_remote_code=False,
        )
        parity = verify_parity(evaluator, reloaded, fast)
        artifacts = RunArtifacts(target.name, temporary)
        artifacts.write_json("verification.json", parity)
        artifacts.write_json("native_checkpoint.json", evaluator.metadata)
        artifacts.write_json(
            "native_tokenizer_manifest.json", evaluator.tokenizer.manifest
        )
        artifacts.write_text(
            "README.md",
            (
                "# Native HF Export\n\n"
                "Standard Hugging Face weights, not a new training run.\n\n"
                f"Source: {evaluator.identity['run_id']}/"
                f"{evaluator.identity['checkpoint_id']}\n\n"
                "Load with AutoModelForCausalLM and AutoTokenizer, "
                "trust_remote_code=False. See verification.json for CPU parity.\n\n"
                "Base models continue text; a chat template does not imply "
                "instruction-following ability. Historical provenance is not "
                "rewritten.\n"
            ),
        )
        license_path = source / "LICENSE"
        if license_path.is_file():
            shutil.copyfile(license_path, temporary / "LICENSE")
        write_hf_manifest(
            temporary,
            {
                "kind": "native-export",
                "model_route": "native",
                "stage": evaluator.metadata.stage.value,
                "source": evaluator.identity,
                "created_at": utc_now(),
                "transformers_version": TRANSFORMERS_VERSION,
                "architecture": hf_config.model_type,
            },
        )
        result = verify_hf_assets(temporary)
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result
