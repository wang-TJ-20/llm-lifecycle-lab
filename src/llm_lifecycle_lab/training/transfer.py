"""Qwen SFT with the same data, objective, budget and resume engine as Native."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from itertools import zip_longest
from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab.artifacts import ArtifactStore
from llm_lifecycle_lab.config import config_sha256
from llm_lifecycle_lab.contracts import ModelRoute, RunConfig, Stage
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.sft import collate_sft, load_sft_splits
from llm_lifecycle_lab.evaluation.suite import read_json
from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError
from llm_lifecycle_lab.interop.transfer import load_transfer_bundle
from llm_lifecycle_lab.provenance import capture_runtime_provenance
from llm_lifecycle_lab.training.batching import DeterministicBatchStream
from llm_lifecycle_lab.training.checkpoint import CheckpointManager
from llm_lifecycle_lab.training.engine import EngineConfig, TrainingEngine
from llm_lifecycle_lab.training.sft import SFTObjective, SFTRun


def run_transfer_sft(
    config: RunConfig,
    *,
    run_id: str | None = None,
    resume_run: str | None = None,
    resume_checkpoint: str | Path | None = None,
    workdir: str | Path = ".",
    metric_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> SFTRun:
    if (
        config.stage is not Stage.SFT
        or config.model_route is not ModelRoute.QWEN3_TRANSFER
    ):
        raise ConfigError("Transfer training requires qwen3-transfer and stage=sft")
    if run_id and resume_run:
        raise ConfigError("run_id and resume_run are mutually exclusive")
    if resume_checkpoint and not resume_run:
        raise ConfigError("resume_checkpoint requires resume_run")
    root = Path(workdir).resolve()

    def path(value) -> Path:
        if not value:
            raise ConfigError("Transfer paths cannot be empty")
        result = Path(value)
        return result if result.is_absolute() else root / result

    snapshot = path(config.model.get("snapshot"))
    manifest = path(config.data.get("manifest"))
    suite = path(config.data.get("evaluation_suite"))
    torch.manual_seed(config.seed)
    bundle, binding = load_transfer_bundle(snapshot, dict(config.model))
    engine_config = EngineConfig.from_dict(config.training)
    if engine_config.sequence_length > bundle.model.config.max_sequence_length:
        raise ConfigError("SFT sequence_length exceeds Qwen context")
    if engine_config.eval_batches * engine_config.micro_batch_size < 2:
        raise ConfigError("Transfer dev budget must cover both languages")
    splits, summary = load_sft_splits(
        manifest,
        tokenizer=bundle.tokenizer,
        sequence_length=engine_config.sequence_length,
        evaluation_suite=suite,
    )
    initialization = {
        "mode": "new-stage",
        "binding": binding,
        "data_manifest_sha256": sha256_file(manifest),
        "evaluation_suite_sha256": summary["suite_sha256"],
        "optimizer_inherited": False,
    }
    engine_config, budget = engine_config.resolve_budget(
        examples_per_epoch=len(splits["train"]),
        supervised_tokens_per_epoch=summary["splits"]["train"]["supervised_tokens"],
    )
    collate = partial(collate_sft, pad_token_id=bundle.tokenizer.pad_token_id)
    stream = DeterministicBatchStream(
        splits["train"],
        batch_size=engine_config.micro_batch_size,
        seed=config.seed,
        collate_fn=collate,
    )
    buckets = [
        [x for x in splits["dev"] if int(x["language_ids"][0]) == lang]
        for lang in (0, 1)
    ]
    ordered = [
        item for pair in zip_longest(*buckets) for item in pair if item is not None
    ]
    ordered = ordered[: engine_config.eval_batches * engine_config.micro_batch_size]
    evaluation = [
        collate(ordered[start : start + engine_config.micro_batch_size])
        for start in range(0, len(ordered), engine_config.micro_batch_size)
    ]
    store = ArtifactStore(path(config.output_dir))
    if resume_run:
        artifacts = store.open_run(config, run_id=resume_run)
        if read_json(artifacts.path / "initialization.json") != initialization:
            raise ArtifactError(
                "Transfer resume inputs/base/adapter configuration changed"
            )
        checkpoint = (
            resume_checkpoint
            or read_json(artifacts.path / "latest_checkpoint.json")["path"]
        )
    else:
        artifacts = store.create_run(config, run_id=run_id)
        checkpoint = None
        artifacts.write_json("initialization.json", initialization)
        artifacts.write_json("training_budget.json", budget.to_dict())
        artifacts.write_json("sft_data_summary.json", summary)
        artifacts.write_json("model_metadata.json", bundle.metadata)
        artifacts.write_json(
            "trainable_parameters.json",
            {
                "names": [name for name, _ in bundle.model.trainable_parameters()],
                "trainable": sum(
                    p.numel() for _, p in bundle.model.trainable_parameters()
                ),
                "total": bundle.metadata.parameter_count,
            },
        )
    manager = CheckpointManager(
        artifacts,
        model_route=ModelRoute.QWEN3_TRANSFER,
        stage=Stage.SFT,
        tokenizer_sha256=bundle.tokenizer.manifest.content_sha256,
        config_sha256=config_sha256(config),
    )
    engine = TrainingEngine(
        config=engine_config,
        budget=budget,
        artifacts=artifacts,
        checkpoint_manager=manager,
        seed=config.seed,
    )
    if not resume_run:
        artifacts.write_json(
            "runtime_environment.json",
            capture_runtime_provenance(workdir=root, device=engine.device),
        )
    result = engine.train(
        bundle=bundle,
        objective=SFTObjective(),
        train_stream=stream,
        evaluation_batches=evaluation,
        resume_from=checkpoint,
        metric_callback=metric_callback,
    )
    return SFTRun(artifacts, result, bundle.metadata.parameter_count, budget)
