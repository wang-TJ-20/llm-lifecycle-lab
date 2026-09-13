"""HF tensor adapter and a stable project chat protocol for Qwen Base models."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from llm_lifecycle_lab.artifacts import RunArtifacts
from llm_lifecycle_lab.evaluation.suite import digest, read_json
from llm_lifecycle_lab.exceptions import ArtifactError, ContractError
from llm_lifecycle_lab.interop.assets import require_hf
from llm_lifecycle_lab.model.protocol import (
    GenerationConfig,
    GenerationOutput,
    ModelOutput,
)
from llm_lifecycle_lab.tokenizer.native import (
    CHAT_END_TOKEN,
    CHAT_START_TOKEN,
    NATIVE_CHAT_TEMPLATE,
    NATIVE_CHAT_TEMPLATE_VERSION,
    render_native_chat,
)


class HFTokenizer:
    def __init__(self, tokenizer, *, content_sha256: str) -> None:
        if not tokenizer.is_fast:
            raise ContractError("HF integration requires a fast tokenizer")
        self.raw = tokenizer
        self.manifest = SimpleNamespace(
            content_sha256=content_sha256,
            source_data_sha256=None,
            revision=content_sha256,
        )
        self.vocab_size = len(tokenizer)
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else self.eos_token_id
        )
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else self.eos_token_id
        )
        vocab = tokenizer.get_vocab()
        self.chat_start_token_id = vocab.get(CHAT_START_TOKEN)
        self.chat_end_token_id = vocab.get(CHAT_END_TOKEN)
        if any(
            x is None
            for x in (
                self.eos_token_id,
                self.bos_token_id,
                self.chat_start_token_id,
                self.chat_end_token_id,
            )
        ):
            raise ContractError(
                "HF tokenizer is missing required text/chat control IDs"
            )
        self.chat_template = NATIVE_CHAT_TEMPLATE
        self.chat_template_version = NATIVE_CHAT_TEMPLATE_VERSION

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False):
        ids = self.raw.encode(text, add_special_tokens=False)
        return (
            ([self.bos_token_id] if add_bos else [])
            + ids
            + ([self.eos_token_id] if add_eos else [])
        )

    def decode(self, ids, *, skip_special_tokens: bool = True) -> str:
        return self.raw.decode(
            list(ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def encode_chat(self, messages, *, add_generation_prompt: bool = False):
        return self.encode(
            render_native_chat(
                messages,
                add_generation_prompt=add_generation_prompt,
            )
        )


class HFModelAdapter:
    """Keep Transformers/PEFT details outside the shared training engine."""

    def __init__(self, model, *, binding: dict[str, Any]) -> None:
        self.raw = model
        self.binding = binding
        self.config = SimpleNamespace(
            max_sequence_length=model.config.max_position_embeddings,
            vocab_size=model.config.vocab_size,
        )

    def forward(
        self,
        *,
        input_ids,
        attention_mask=None,
        use_cache=False,
        cache=None,
    ) -> ModelOutput:
        result = self.raw(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=cache,
        )
        return ModelOutput(logits=result.logits, cache=result.past_key_values)

    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids,
        attention_mask,
        config: GenerationConfig,
    ) -> GenerationOutput:
        if input_ids.shape[1] + config.max_new_tokens > self.config.max_sequence_length:
            raise ContractError("HF generation exceeds max_sequence_length")
        was_training = self.raw.training
        self.raw.eval()
        cuda_devices = (
            [input_ids.device.index or 0] if input_ids.device.type == "cuda" else []
        )
        mps_state = (
            torch.mps.get_rng_state() if input_ids.device.type == "mps" else None
        )
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(config.seed)
                options = {}
                if config.do_sample:
                    options = {
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": 0,
                    }
                ids = self.raw.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                    if attention_mask is not None
                    else torch.ones_like(input_ids),
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    eos_token_id=config.eos_token_id,
                    pad_token_id=config.pad_token_id,
                    **options,
                )
        finally:
            self.raw.train(was_training)
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
        count = ids.shape[1] - input_ids.shape[1]
        ended = config.eos_token_id is not None and bool(
            (ids[:, -1] == config.eos_token_id).all()
        )
        return GenerationOutput(
            token_ids=ids,
            prompt_tokens=input_ids.shape[1],
            generated_tokens=count,
            stop_reason="eos" if ended else "length",
        )

    def trainable_parameters(self):
        return tuple(
            (name, value)
            for name, value in self.raw.named_parameters()
            if value.requires_grad
        )

    def to_device(self, device) -> None:
        self.raw.to(device)

    def set_training(self, training: bool) -> None:
        self.raw.train(training)

    def is_training(self) -> bool:
        return self.raw.training

    def save(self, path: Path) -> None:
        if path.exists() and any(path.iterdir()):
            raise ArtifactError("HF checkpoint target is not empty")
        self.raw.save_pretrained(path, safe_serialization=True)
        RunArtifacts(path.name, path).write_json("model_binding.json", self.binding)

    def load(self, path: Path) -> None:
        if read_json(path / "model_binding.json") != self.binding:
            raise ArtifactError("HF checkpoint base/adapter binding mismatch")
        if self.binding["training_method"] == "lora":
            require_hf(peft=True)
            from peft import load_peft_weights, set_peft_model_state_dict

            state = load_peft_weights(str(path), device="cpu")
            result = set_peft_model_state_dict(self.raw, state)
            if result.unexpected_keys:
                raise ArtifactError("unexpected LoRA checkpoint weights")
        else:
            hf = require_hf()
            restored = hf.AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
            )
            self.raw.load_state_dict(restored.state_dict(), strict=True)


def apply_lora(model, settings: dict[str, Any]):
    require_hf(peft=True)
    from peft import LoraConfig, get_peft_model

    if set(settings) - {"rank", "alpha", "dropout", "target_modules"}:
        raise ContractError("unknown LoRA setting")
    rank = settings.get("rank", 8)
    alpha = settings.get("alpha", 16)
    dropout = settings.get("dropout", 0.0)
    targets = settings.get("target_modules", ["q_proj", "v_proj"])
    if (
        type(rank) is not int
        or rank <= 0
        or type(alpha) is not int
        or alpha <= 0
        or not isinstance(dropout, (float, int))
        or not 0 <= dropout < 1
        or not isinstance(targets, (list, tuple))
        or not targets
        or set(targets)
        - {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    ):
        raise ContractError("invalid LoRA settings")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(targets),
            bias="none",
        ),
    )
    if any(
        p.requires_grad and "lora_" not in name for name, p in model.named_parameters()
    ):
        raise ContractError("LoRA unexpectedly left base parameters trainable")
    return model


def binding_digest(binding: dict[str, Any]) -> str:
    return digest(binding)
