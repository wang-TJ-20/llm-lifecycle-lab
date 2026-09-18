"""End-to-end orchestration for the native pretraining stage."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab.artifacts import ArtifactStore, RunArtifacts
from llm_lifecycle_lab.config import config_sha256
from llm_lifecycle_lab.contracts import (
    CheckpointMetadata,
    EvaluationReport,
    ModelManifest,
    ModelMetadata,
    ModelRoute,
    RunConfig,
    Stage,
)
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.packing import (
    DiskPackedPretrainingDataset,
    collate_pretraining_batch,
    load_packed_pretraining_manifest,
)
from llm_lifecycle_lab.data.prepare import load_data_manifest
from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError, ContractError
from llm_lifecycle_lab.model.bundle import ModelBundle
from llm_lifecycle_lab.model.native import (
    NativeTransformer,
    load_native_model_config,
)
from llm_lifecycle_lab.provenance import capture_runtime_provenance
from llm_lifecycle_lab.tokenizer import NativeTokenizer
from llm_lifecycle_lab.training.batching import DeterministicBatchStream
from llm_lifecycle_lab.training.checkpoint import CheckpointManager
from llm_lifecycle_lab.training.engine import (
    EngineConfig,
    ResolvedTrainingBudget,
    TrainingEngine,
    TrainingResult,
    evaluate_objective,
    resolve_device,
)
from llm_lifecycle_lab.training.stages import PretrainObjective


@dataclass(frozen=True, slots=True)
class PretrainingRun:
    artifacts: RunArtifacts
    result: TrainingResult
    parameter_count: int
    budget: ResolvedTrainingBudget


def evaluate_native_pretraining(
    config: RunConfig,
    *,
    checkpoint: str | Path,
    split: str = "dev",
    workdir: str | Path = ".",
) -> dict[str, Any]:
    if (
        config.stage is not Stage.PRETRAIN
        or config.model_route is not ModelRoute.NATIVE
    ):
        raise ConfigError("pretrain evaluation requires a native pretrain config")

    root = Path(workdir).resolve()
    model_config_path = _required_path(config.model, "config", root=root)
    tokenizer_path = _required_path(config.model, "tokenizer", root=root)
    data_manifest_path = _required_path(config.data, "manifest", root=root)
    packed_manifest_path = _required_path(
        config.data,
        "packed_manifest",
        root=root,
    )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path

    engine_config = EngineConfig.from_dict(config.training)
    model_config = load_native_model_config(model_config_path)
    tokenizer = NativeTokenizer.from_directory(tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ConfigError("tokenizer and model vocabulary sizes do not match")

    metadata = _checkpoint_metadata(checkpoint_path)
    expected = {
        "model_route": (metadata.model_route, config.model_route),
        "stage": (metadata.stage, Stage.PRETRAIN),
        "tokenizer_sha256": (
            metadata.tokenizer_sha256,
            tokenizer.manifest.content_sha256,
        ),
        "config_sha256": (metadata.config_sha256, config_sha256(config)),
    }
    mismatches = [
        field for field, (actual, wanted) in expected.items() if actual != wanted
    ]
    if mismatches:
        raise ArtifactError(
            "evaluation checkpoint is incompatible: " + ", ".join(mismatches)
        )

    model = NativeTransformer(model_config)
    model.load(checkpoint_path / "model")
    device = resolve_device(engine_config.device)
    model.to_device(device)
    batches = _batches_for_split(
        data_manifest_path,
        split=split,
        tokenizer=tokenizer,
        engine_config=engine_config,
        packed_manifest_path=packed_manifest_path,
    )
    metrics = evaluate_objective(
        model=model,
        objective=PretrainObjective(),
        batches=batches,
        device=device,
        dtype=engine_config.torch_dtype,
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": metadata.step,
        "split": split,
        **metrics,
    }
    run_path = checkpoint_path.parent.parent
    if run_path.name == metadata.run_id and (run_path / "run_manifest.json").is_file():
        evaluation = EvaluationReport(
            run_id=metadata.run_id,
            suite=f"pretrain-{split}",
            metrics=metrics,
            sample_count=sum(int(batch["input_ids"].shape[0]) for batch in batches),
        )
        report_path = RunArtifacts(
            run_id=metadata.run_id,
            path=run_path,
        ).write_json(
            f"evaluations/pretrain-{split}-step-{metadata.step:08d}.json",
            evaluation,
        )
        result["report_path"] = str(report_path)
    return result


def run_native_pretraining(
    config: RunConfig,
    *,
    run_id: str | None = None,
    resume_run: str | None = None,
    resume_checkpoint: str | Path | None = None,
    workdir: str | Path = ".",
    metric_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> PretrainingRun:
    if config.stage is not Stage.PRETRAIN:
        raise ConfigError("native pretraining requires stage=pretrain")
    if config.model_route is not ModelRoute.NATIVE:
        raise ConfigError("pretraining is only available for model_route=native")
    if run_id is not None and resume_run is not None:
        raise ConfigError("run_id and resume_run are mutually exclusive")
    if resume_checkpoint is not None and resume_run is None:
        raise ConfigError("resume_checkpoint requires resume_run")

    root = Path(workdir).resolve()
    model_config_path = _required_path(
        config.model,
        "config",
        root=root,
    )
    tokenizer_path = _required_path(
        config.model,
        "tokenizer",
        root=root,
    )
    data_manifest_path = _required_path(
        config.data,
        "manifest",
        root=root,
    )
    packed_manifest_path = _required_path(
        config.data,
        "packed_manifest",
        root=root,
    )
    engine_config = EngineConfig.from_dict(config.training)
    model_config = load_native_model_config(model_config_path)
    if model_config.model_id != config.model["model_id"]:
        raise ConfigError("pipeline model_id does not match native model config")
    if engine_config.sequence_length > model_config.max_sequence_length:
        raise ConfigError("training.sequence_length exceeds model max_sequence_length")

    tokenizer = NativeTokenizer.from_directory(tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ConfigError(
            f"tokenizer vocab_size {tokenizer.vocab_size} does not match "
            f"model vocab_size {model_config.vocab_size}"
        )
    data_manifest_sha256 = sha256_file(data_manifest_path)

    torch.manual_seed(config.seed)
    model = NativeTransformer(model_config)
    metadata = ModelMetadata(
        model_route=config.model_route,
        provider="native",
        model_id=model_config.model_id,
        architecture="dense-decoder",
        tokenizer_revision=tokenizer.manifest.revision,
        chat_template_version=tokenizer.chat_template_version,
        parameter_count=model.parameter_count,
        capabilities=("forward", "generate", "kv-cache", "full-training"),
    )
    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        chat_template=None,
        metadata=metadata,
    )

    train_dataset = _load_pretraining_dataset(
        data_manifest_path,
        split="train",
        tokenizer=tokenizer,
        sequence_length=engine_config.sequence_length,
        packed_manifest_path=packed_manifest_path,
    )
    engine_config, training_budget = engine_config.resolve_budget(
        examples_per_epoch=train_dataset.stats.examples,
        supervised_tokens_per_epoch=train_dataset.stats.supervised_tokens,
    )
    train_stream = DeterministicBatchStream(
        train_dataset,
        batch_size=engine_config.micro_batch_size,
        seed=config.seed,
        collate_fn=collate_pretraining_batch,
    )
    evaluation_batches = _evaluation_batches(
        data_manifest_path,
        tokenizer=tokenizer,
        engine_config=engine_config,
        packed_manifest_path=packed_manifest_path,
    )

    output_root = Path(config.output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    store = ArtifactStore(output_root)
    if resume_run is not None:
        artifacts = store.open_run(config, run_id=resume_run)
        checkpoint = resume_checkpoint or _latest_checkpoint(artifacts)
    else:
        artifacts = store.create_run(config, run_id=run_id)
        checkpoint = None
        artifacts.write_json("tokenizer_manifest.json", tokenizer.manifest)
        artifacts.write_text(
            "tokenizer/tokenizer.json",
            (tokenizer_path / "tokenizer.json").read_text(encoding="utf-8"),
        )
        artifacts.write_json("model_config.json", asdict(model_config))
        artifacts.write_json(
            "model_manifest.json",
            ModelManifest(
                metadata=metadata,
                config_sha256=sha256_file(model_config_path),
            ),
        )
        artifacts.write_json(
            "data_snapshot.json",
            load_data_manifest(data_manifest_path),
        )
        artifacts.write_json(
            "data_provenance.json",
            {
                "training_data_manifest": str(data_manifest_path),
                "training_data_manifest_sha256": data_manifest_sha256,
                "tokenizer_source_data_manifest_sha256": (
                    tokenizer.manifest.source_data_sha256
                ),
                "tokenizer_sha256": tokenizer.manifest.content_sha256,
            },
        )
        artifacts.write_json(
            "packed_data_snapshot.json",
            load_packed_pretraining_manifest(packed_manifest_path),
        )
        artifacts.write_json("training_budget.json", training_budget.to_dict())

    if metric_callback is not None:
        metric_callback(
            {
                "event": "training-plan",
                **training_budget.to_dict(),
            }
        )

    manager = CheckpointManager(
        artifacts,
        model_route=config.model_route,
        stage=Stage.PRETRAIN,
        tokenizer_sha256=tokenizer.manifest.content_sha256,
        config_sha256=config_sha256(config),
    )
    engine = TrainingEngine(
        config=engine_config,
        budget=training_budget,
        artifacts=artifacts,
        checkpoint_manager=manager,
        seed=config.seed,
    )
    if resume_run is None:
        artifacts.write_json(
            "runtime_environment.json",
            capture_runtime_provenance(
                workdir=root,
                device=engine.device,
            ),
        )
    result = engine.train(
        bundle=bundle,
        objective=PretrainObjective(),
        train_stream=train_stream,
        evaluation_batches=evaluation_batches,
        resume_from=checkpoint,
        metric_callback=metric_callback,
    )
    return PretrainingRun(
        artifacts=artifacts,
        result=result,
        parameter_count=model.parameter_count,
        budget=training_budget,
    )


def _required_path(
    mapping: object,
    field: str,
    *,
    root: Path,
) -> Path:
    if not hasattr(mapping, "get"):
        raise ConfigError(f"configuration section for {field} must be a mapping")
    value = mapping.get(field)
    if not value:
        raise ConfigError(f"configuration must declare {field}")
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _load_pretraining_dataset(
    manifest_path: Path,
    *,
    split: str,
    tokenizer: NativeTokenizer,
    sequence_length: int,
    packed_manifest_path: Path,
) -> DiskPackedPretrainingDataset:
    return DiskPackedPretrainingDataset.from_manifest(
        packed_manifest_path,
        split=split,
        data_manifest_sha256=sha256_file(manifest_path),
        tokenizer_sha256=tokenizer.manifest.content_sha256,
        sequence_length=sequence_length,
    )


def _evaluation_batches(
    manifest_path: Path,
    *,
    tokenizer: NativeTokenizer,
    engine_config: EngineConfig,
    packed_manifest_path: Path,
) -> list[dict[str, torch.Tensor]]:
    return _batches_for_split(
        manifest_path,
        split="dev",
        tokenizer=tokenizer,
        engine_config=engine_config,
        packed_manifest_path=packed_manifest_path,
    )


def _batches_for_split(
    manifest_path: Path,
    *,
    split: str,
    tokenizer: NativeTokenizer,
    engine_config: EngineConfig,
    packed_manifest_path: Path,
) -> list[dict[str, torch.Tensor]]:
    manifest = load_data_manifest(manifest_path)
    try:
        split_manifest = next(item for item in manifest.splits if item.name == split)
    except StopIteration as exc:
        raise ConfigError(f"unknown evaluation split: {split}") from exc
    if split_manifest.records == 0:
        raise ConfigError(f"pretraining evaluation requires a non-empty {split} split")
    dataset = _load_pretraining_dataset(
        manifest_path,
        split=split,
        tokenizer=tokenizer,
        sequence_length=engine_config.sequence_length,
        packed_manifest_path=packed_manifest_path,
    )
    indices = _stratified_evaluation_indices(
        dataset,
        max_examples=(engine_config.eval_batches * engine_config.micro_batch_size),
    )
    batches = []
    for start in range(0, len(indices), engine_config.micro_batch_size):
        examples = [
            dataset[index]
            for index in indices[start : start + engine_config.micro_batch_size]
        ]
        batches.append(collate_pretraining_batch(examples))
        if len(batches) >= engine_config.eval_batches:
            break
    return batches


def _stratified_evaluation_indices(
    dataset: DiskPackedPretrainingDataset,
    *,
    max_examples: int,
) -> list[int]:
    language_buckets: dict[int, list[int]] = {0: [], 1: []}
    unlabelled: list[int] = []
    for index in range(len(dataset)):
        example = dataset[index]
        valid = example["labels"] != -100
        language_ids = example["language_ids"][valid]
        counts = {
            language_id: int((language_ids == language_id).sum())
            for language_id in language_buckets
        }
        best_language = max(counts, key=counts.get)
        if counts[best_language] == 0:
            unlabelled.append(index)
        else:
            language_buckets[best_language].append(index)

    selected: list[int] = []
    positions = {language_id: 0 for language_id in language_buckets}
    while len(selected) < max_examples:
        added = False
        for language_id, bucket in language_buckets.items():
            position = positions[language_id]
            if position < len(bucket) and len(selected) < max_examples:
                selected.append(bucket[position])
                positions[language_id] += 1
                added = True
        if not added:
            break

    selected_set = set(selected)
    remaining = (
        index
        for index in chain(unlabelled, range(len(dataset)))
        if index not in selected_set
    )
    selected.extend(
        index
        for _, index in zip(
            range(max_examples - len(selected)),
            remaining,
            strict=False,
        )
    )
    return selected


def _checkpoint_metadata(path: Path) -> CheckpointMetadata:
    try:
        value = json.loads(
            (path / "checkpoint_metadata.json").read_text(encoding="utf-8")
        )
        return CheckpointMetadata.from_dict(value)
    except (
        OSError,
        ContractError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ArtifactError(f"invalid checkpoint metadata under {path}: {exc}") from exc


def _latest_checkpoint(artifacts: RunArtifacts) -> str:
    path = artifacts.artifact_path("latest_checkpoint.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return str(value["path"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ConfigError(
            f"cannot resolve latest checkpoint for {artifacts.run_id}: {exc}"
        ) from exc
