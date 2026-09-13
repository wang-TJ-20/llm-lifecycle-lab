"""Load frozen Native inputs and expose generation and conditional likelihood."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from llm_lifecycle_lab.contracts import CheckpointMetadata, ModelRoute
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.evaluation.suite import read_json
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.native import NativeTransformer, load_native_model_config
from llm_lifecycle_lab.model.protocol import GenerationConfig
from llm_lifecycle_lab.tokenizer import NativeTokenizer
from llm_lifecycle_lab.tokenizer.native import SPECIAL_TOKENS
from llm_lifecycle_lab.training.engine import resolve_device


class NativeEvaluator:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        tokenizer_dir: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        root = Path(checkpoint).resolve()
        metadata_path = root / "checkpoint_metadata.json"
        if (root / "release_manifest.json").is_file():
            # Import lazily: publication consumes evaluation, not vice versa at import.
            from llm_lifecycle_lab.release import verify_release_files

            verify_release_files(root)
            metadata_path = root / "metadata/checkpoint_metadata.json"
        self.metadata = CheckpointMetadata.from_dict(read_json(metadata_path))
        if self.metadata.model_route is not ModelRoute.NATIVE:
            raise ContractError("capability evaluation currently supports Native only")
        selected_tokenizer = (
            Path(tokenizer_dir) if tokenizer_dir else root / "tokenizer"
        )
        self.tokenizer = NativeTokenizer.from_directory(selected_tokenizer)
        if self.metadata.tokenizer_sha256 != self.tokenizer.manifest.content_sha256:
            raise ContractError("checkpoint and evaluation tokenizer hashes differ")
        self.config = load_native_model_config(root / "model/config.json")
        if self.config.vocab_size != self.tokenizer.vocab_size:
            raise ContractError("model and tokenizer vocabularies differ")
        self.identity = {
            "checkpoint_id": self.metadata.checkpoint_id,
            "run_id": self.metadata.run_id,
            "stage": self.metadata.stage.value,
            "weights_sha256": sha256_file(root / "model/model.pt"),
            "config_sha256": sha256_file(root / "model/config.json"),
            "tokenizer_sha256": self.tokenizer.manifest.content_sha256,
        }
        self.device = resolve_device(device)
        self.model = NativeTransformer(self.config)
        self.model.load(root / "model")
        self.model.to_device(self.device)
        self.model.set_training(False)

    def prompt_ids(self, messages: list[dict[str, str]], protocol: str) -> list[int]:
        if protocol == "native-chat-v1":
            # Generated text can contain control strings or be empty. They are kept
            # in reports, but escaped before putting them back into the next turn.
            safe = []
            for message in messages:
                content = message["content"] or "[empty response]"
                for token in SPECIAL_TOKENS:
                    content = content.replace(token, "[control token]")
                safe.append({**message, "content": content})
            return self.tokenizer.encode_chat(safe, add_generation_prompt=True)
        if protocol != "plain-v1":
            raise ContractError("unknown evaluation prompt protocol")
        text = (
            "".join(
                f"{message['role'].capitalize()}: {message['content']}\n"
                for message in messages
            )
            + "Assistant:"
        )
        return self.tokenizer.encode(text, add_bos=True)

    @torch.inference_mode()
    def generate(
        self, ids: list[int], *, max_new_tokens: int, protocol: str = "raw"
    ) -> dict[str, Any]:
        if len(ids) + max_new_tokens > self.config.max_sequence_length:
            raise ContractError("evaluation context overflow; refusing to truncate")
        end = (
            self.tokenizer.chat_end_token_id
            if protocol == "native-chat-v1"
            else self.tokenizer.eos_token_id
        )
        output = self.model.generate(
            input_ids=torch.tensor([ids], device=self.device),
            attention_mask=None,
            config=GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                seed=42,
                eos_token_id=end,
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )
        tokens = output.token_ids[0, len(ids) :].tolist()
        content = tokens[:-1] if tokens and tokens[-1] == end else tokens
        return {
            "text": self.tokenizer.decode(content, skip_special_tokens=False),
            "token_ids": content,
            "stop_reason": output.stop_reason,
        }

    @torch.inference_mode()
    def nll(self, prefix: list[int], target: list[int]) -> float:
        if not prefix or not target:
            raise ContractError("conditional likelihood needs prefix and target tokens")
        ids = prefix + target
        if len(ids) > self.config.max_sequence_length:
            raise ContractError("likelihood context overflow; refusing to truncate")
        tensor = torch.tensor([ids], device=self.device)
        logits = self.model.forward(input_ids=tensor, use_cache=False).logits
        loss = (
            F.cross_entropy(
                logits[0, len(prefix) - 1 : -1].float(),
                tensor[0, len(prefix) :],
                reduction="none",
            )
            .cpu()
            .double()
            .sum()
        )
        if not torch.isfinite(loss):
            raise ContractError("non-finite evaluation likelihood")
        return float(loss)
