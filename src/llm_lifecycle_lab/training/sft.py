"""Initialize SFT from a Base model, or resume the exact same SFT run."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from itertools import zip_longest
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.artifacts import ArtifactStore, RunArtifacts
from llm_lifecycle_lab.config import config_sha256
from llm_lifecycle_lab.contracts import (
    CheckpointMetadata,
    ModelMetadata,
    ModelRoute,
    RunConfig,
    Stage,
)
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.sft import collate_sft, load_sft_splits
from llm_lifecycle_lab.evaluation.suite import read_json
from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError
from llm_lifecycle_lab.model.bundle import ModelBundle
from llm_lifecycle_lab.model.native import NativeTransformer, load_native_model_config
from llm_lifecycle_lab.provenance import capture_runtime_provenance
from llm_lifecycle_lab.tokenizer import NativeTokenizer
from llm_lifecycle_lab.training.batching import DeterministicBatchStream
from llm_lifecycle_lab.training.checkpoint import CheckpointManager
from llm_lifecycle_lab.training.engine import (
    EngineConfig,
    ResolvedTrainingBudget,
    TrainingEngine,
    TrainingResult,
)
from llm_lifecycle_lab.training.stages import PretrainObjective


class SFTObjective(PretrainObjective):
    """Same next-token objective; the dataset supplies assistant-only labels."""

    name = "sft"


@dataclass(frozen=True, slots=True)
class SFTRun:
    artifacts: RunArtifacts
    result: TrainingResult
    parameter_count: int
    budget: ResolvedTrainingBudget


def run_native_sft(
    config: RunConfig,
    *,
    run_id: str | None = None,
    resume_run: str | None = None,
    resume_checkpoint: str | Path | None = None,
    workdir: str | Path = ".",
    metric_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> SFTRun:
    if config.stage is not Stage.SFT or config.model_route is not ModelRoute.NATIVE:
        raise ConfigError("SFT requires stage=sft and model_route=native")
    if run_id and resume_run:
        raise ConfigError("run_id and resume_run are mutually exclusive")
    if resume_checkpoint and not resume_run:
        raise ConfigError("resume_checkpoint requires resume_run")
    root = Path(workdir).resolve()

    def required(mapping: Mapping[str, Any], name: str) -> Path:
        if not mapping.get(name):
            raise ConfigError(f"SFT configuration must declare {name}")
        path = Path(mapping[name])
        return path if path.is_absolute() else root / path

    parent = required(config.model, "init_checkpoint")
    tokenizer_dir = required(config.model, "tokenizer")
    manifest_path = required(config.data, "manifest")
    suite_path = required(config.data, "evaluation_suite")
    metadata_path = parent / "checkpoint_metadata.json"
    waiver = None
    if (parent / "release_manifest.json").is_file():
        from llm_lifecycle_lab.release import verify_release_files

        verify_release_files(parent)
        metadata_path = parent / "metadata/checkpoint_metadata.json"
        waiver = read_json(parent / "release_manifest.json").get("provenance_status")
    parent_metadata = CheckpointMetadata.from_dict(read_json(metadata_path))
    if (
        parent_metadata.stage is not Stage.PRETRAIN
        or parent_metadata.model_route is not ModelRoute.NATIVE
    ):
        raise ConfigError(
            "new SFT stages must initialize from a Native Base checkpoint"
        )
    tokenizer = NativeTokenizer.from_directory(tokenizer_dir)
    if parent_metadata.tokenizer_sha256 != tokenizer.manifest.content_sha256:
        raise ConfigError("Base checkpoint and SFT tokenizer hashes differ")
    model_config = load_native_model_config(parent / "model/config.json")
    if (
        model_config.vocab_size != tokenizer.vocab_size
        or model_config.model_id != config.model["model_id"]
    ):
        raise ConfigError("SFT pipeline model does not match its Base checkpoint")
    engine_config = EngineConfig.from_dict(config.training)
    if engine_config.sequence_length > model_config.max_sequence_length:
        raise ConfigError("SFT sequence_length exceeds model context")
    splits, summary = load_sft_splits(
        manifest_path,
        tokenizer=tokenizer,
        sequence_length=engine_config.sequence_length,
        evaluation_suite=suite_path,
    )
    initialization = {
        "mode": "weights-only-new-stage",
        "parent_checkpoint": parent_metadata.to_dict(),
        "parent_weights_sha256": sha256_file(parent / "model/model.pt"),
        "parent_model_config_sha256": sha256_file(parent / "model/config.json"),
        "parent_provenance_status": waiver,
        "data_manifest_sha256": sha256_file(manifest_path),
        "evaluation_suite_sha256": summary["suite_sha256"],
        "optimizer_inherited": False,
        "initial_step": 0,
    }
    model = NativeTransformer(model_config)
    model.load(parent / "model")
    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
        metadata=ModelMetadata(
            model_route=ModelRoute.NATIVE,
            provider="native",
            model_id=model_config.model_id,
            architecture="dense-decoder",
            tokenizer_revision=tokenizer.manifest.revision,
            chat_template_version=tokenizer.chat_template_version,
            parameter_count=model.parameter_count,
            capabilities=("forward", "generate", "kv-cache", "full-training"),
        ),
    )
    collate = partial(collate_sft, pad_token_id=tokenizer.pad_token_id)
    train = splits["train"]
    engine_config, budget = engine_config.resolve_budget(
        examples_per_epoch=len(train),
        supervised_tokens_per_epoch=summary["splits"]["train"]["supervised_tokens"],
    )
    stream = DeterministicBatchStream(
        train,
        batch_size=engine_config.micro_batch_size,
        seed=config.seed,
        collate_fn=collate,
    )
    # Alternate languages in the bounded dev subset, rather than accepting a
    # first-N subset that accidentally omits one language.
    buckets = [
        [x for x in splits["dev"] if int(x["language_ids"][0]) == language]
        for language in (0, 1)
    ]
    indices = [
        item for pair in zip_longest(*buckets) for item in pair if item is not None
    ]
    if engine_config.eval_batches * engine_config.micro_batch_size < 2:
        raise ConfigError("SFT dev evaluation budget must cover both languages")
    indices = indices[: engine_config.eval_batches * engine_config.micro_batch_size]
    evaluation = [
        collate(indices[start : start + engine_config.micro_batch_size])
        for start in range(0, len(indices), engine_config.micro_batch_size)
    ]
    output_root = Path(config.output_dir)
    store = ArtifactStore(
        output_root if output_root.is_absolute() else root / output_root
    )
    if resume_run:
        artifacts = store.open_run(config, run_id=resume_run)
        if read_json(artifacts.path / "initialization.json") != initialization:
            raise ArtifactError(
                "SFT resume inputs/parent differ from frozen initialization"
            )
        checkpoint = (
            resume_checkpoint
            or read_json(artifacts.path / "latest_checkpoint.json")["path"]
        )
    else:
        artifacts = store.create_run(config, run_id=run_id)
        checkpoint = None
        artifacts.write_json("initialization.json", initialization)
        artifacts.write_json("sft_data_summary.json", summary)
        artifacts.write_json("training_budget.json", budget.to_dict())
        artifacts.write_json("model_config.json", asdict(model_config))
        artifacts.write_json("tokenizer/tokenizer_manifest.json", tokenizer.manifest)
        artifacts.write_text(
            "tokenizer/tokenizer.json",
            (tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"),
        )
    manager = CheckpointManager(
        artifacts,
        model_route=ModelRoute.NATIVE,
        stage=Stage.SFT,
        tokenizer_sha256=tokenizer.manifest.content_sha256,
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
    return SFTRun(artifacts, result, model.parameter_count, budget)
