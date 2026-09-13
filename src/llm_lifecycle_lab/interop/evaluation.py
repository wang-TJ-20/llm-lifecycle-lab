"""The existing capability probes evaluated through a verified HF backend."""

from __future__ import annotations

from pathlib import Path

import torch

from llm_lifecycle_lab.contracts import CheckpointMetadata, ModelRoute
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.evaluation.native import NativeEvaluator
from llm_lifecycle_lab.evaluation.suite import digest, read_json
from llm_lifecycle_lab.exceptions import ArtifactError, ContractError
from llm_lifecycle_lab.interop.assets import (
    payload_fingerprints,
    require_hf,
    verify_hf_assets,
)
from llm_lifecycle_lab.interop.backend import HFModelAdapter, HFTokenizer
from llm_lifecycle_lab.interop.transfer import load_transfer_bundle
from llm_lifecycle_lab.training.engine import resolve_device


class HFEvaluator(NativeEvaluator):
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        base_model: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        hf = require_hf()
        root = Path(checkpoint)
        if (root / "checkpoint_metadata.json").is_file():
            if base_model is None:
                raise ContractError(
                    "HF training checkpoint requires --base-model snapshot"
                )
            metadata = CheckpointMetadata.from_dict(
                read_json(root / "checkpoint_metadata.json")
            )
            if metadata.model_route is not ModelRoute.QWEN3_TRANSFER:
                raise ContractError("use the Native backend for Native checkpoints")
            binding = read_json(root / "model/model_binding.json")
            bundle, _ = load_transfer_bundle(
                base_model,
                {
                    **binding,
                    "model_id": binding["repo_id"],
                    "tokenizer_revision": binding["revision"],
                },
            )
            if (
                sha256_file(Path(base_model) / "hf_manifest.json")
                != (binding["snapshot_manifest_sha256"])
            ):
                raise ArtifactError(
                    "HF checkpoint references a different base snapshot"
                )
            bundle.model.load(root / "model")
            self.model, self.tokenizer = bundle.model, bundle.tokenizer
            if metadata.tokenizer_sha256 != self.tokenizer.manifest.content_sha256:
                raise ArtifactError("HF checkpoint/tokenizer hash mismatch")
            self.identity = {
                "run_id": metadata.run_id,
                "checkpoint_id": metadata.checkpoint_id,
                "stage": metadata.stage.value,
                "model_route": metadata.model_route.value,
                "weights_sha256": digest(
                    {"files": payload_fingerprints(root / "model")}
                ),
                "config_sha256": digest(binding),
                "tokenizer_sha256": self.tokenizer.manifest.content_sha256,
            }
        else:
            manifest = verify_hf_assets(root)
            if manifest["kind"] not in {"native-export", "qwen-snapshot"}:
                raise ContractError("unsupported HF artifact kind")
            model = hf.AutoModelForCausalLM.from_pretrained(
                root,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
            )
            tokenizer = hf.AutoTokenizer.from_pretrained(
                root,
                local_files_only=True,
                trust_remote_code=False,
            )
            self.tokenizer = HFTokenizer(
                tokenizer,
                content_sha256=sha256_file(root / "tokenizer.json"),
            )
            if manifest["kind"] == "native-export":
                self.tokenizer.manifest.source_data_sha256 = read_json(
                    root / "native_tokenizer_manifest.json"
                )["source_data_sha256"]
            self.model = HFModelAdapter(model, binding={"training_method": "full"})
            source = manifest.get("source", {})
            self.identity = {
                "run_id": source.get("run_id", manifest.get("repo_id")),
                "checkpoint_id": source.get("checkpoint_id", manifest.get("revision")),
                "stage": manifest["stage"],
                "model_route": manifest["model_route"],
                "weights_sha256": digest(
                    {
                        "files": [
                            entry
                            for entry in manifest["files"]
                            if entry["path"].endswith(".safetensors")
                        ]
                    }
                ),
                "config_sha256": sha256_file(root / "config.json"),
                "tokenizer_sha256": self.tokenizer.manifest.content_sha256,
            }
        self.config = self.model.config
        self.device = resolve_device(device)
        self.model.to_device(self.device)
        self.model.set_training(False)
