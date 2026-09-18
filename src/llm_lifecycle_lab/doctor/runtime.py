"""One-batch runtime validation for implemented native training stages."""

from __future__ import annotations

from pathlib import Path

from llm_lifecycle_lab.contracts import ModelRoute, RunConfig, Stage
from llm_lifecycle_lab.doctor.result import CheckResult, CheckStatus


def check_real_batch(config: RunConfig, root: Path) -> CheckResult:
    if (
        config.model_route is ModelRoute.QWEN3_TRANSFER
        or config.stage is not Stage.PRETRAIN
    ):
        return CheckResult(
            name="real-batch",
            status=CheckStatus.WARN,
            message="real batch check is not implemented for this route/stage",
            details={
                "model_route": config.model_route.value,
                "stage": config.stage.value,
            },
        )

    try:
        import torch

        from llm_lifecycle_lab.data.fingerprint import sha256_file
        from llm_lifecycle_lab.data.packing import (
            DiskPackedPretrainingDataset,
            collate_pretraining_batch,
        )
        from llm_lifecycle_lab.model.native import (
            NativeTransformer,
            load_native_model_config,
        )
        from llm_lifecycle_lab.tokenizer import NativeTokenizer
        from llm_lifecycle_lab.training.engine import (
            EngineConfig,
            resolve_device,
        )
        from llm_lifecycle_lab.training.stages import PretrainObjective

        model_config_path = _resolve(config.model.get("config"), root)
        tokenizer_path = _resolve(config.model.get("tokenizer"), root)
        data_manifest_path = _resolve(config.data.get("manifest"), root)
        engine_config = EngineConfig.from_dict(config.training)
        model_config = load_native_model_config(model_config_path)
        tokenizer = NativeTokenizer.from_directory(tokenizer_path)
        if tokenizer.vocab_size != model_config.vocab_size:
            raise ValueError("tokenizer and model vocabulary sizes do not match")
        packed_manifest_path = _resolve(
            config.data.get("packed_manifest"),
            root,
        )
        dataset = DiskPackedPretrainingDataset.from_manifest(
            packed_manifest_path,
            split="train",
            data_manifest_sha256=sha256_file(data_manifest_path),
            tokenizer_sha256=tokenizer.manifest.content_sha256,
            sequence_length=engine_config.sequence_length,
        )
        batch = collate_pretraining_batch([dataset[0]])
        device = resolve_device(engine_config.device)
        model = NativeTransformer(model_config)
        model.to_device(device)
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        output = PretrainObjective()(model, batch)
        output.loss.backward()
        finite_gradients = all(
            bool(torch.isfinite(parameter.grad).all())
            for _, parameter in model.trainable_parameters()
            if parameter.grad is not None
        )
        if not finite_gradients:
            raise ValueError("one-batch backward produced non-finite gradients")
        return CheckResult(
            name="real-batch",
            status=CheckStatus.PASS,
            message=(
                f"one-batch forward/backward passed with loss "
                f"{float(output.loss.detach()):.6f}"
            ),
            details={
                "device": str(device),
                "parameters": model.parameter_count,
                "sequence_length": engine_config.sequence_length,
            },
        )
    except Exception as exc:
        return CheckResult(
            name="real-batch",
            status=CheckStatus.FAIL,
            message=f"{type(exc).__name__}: {exc}",
            details={
                "model_route": config.model_route.value,
                "stage": config.stage.value,
            },
        )


def _resolve(value: object, root: Path) -> Path:
    if value is None or not str(value):
        raise ValueError("required runtime path is missing")
    path = Path(str(value))
    return path if path.is_absolute() else root / path
