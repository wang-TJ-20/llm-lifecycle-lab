"""Native on-policy GRPO/RLVR with a frozen parent reference."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
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
from llm_lifecycle_lab.data.grpo import collate_grpo, load_grpo_splits
from llm_lifecycle_lab.evaluation.suite import read_json
from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError
from llm_lifecycle_lab.model.bundle import ModelBundle
from llm_lifecycle_lab.model.native import NativeTransformer, load_native_model_config
from llm_lifecycle_lab.provenance import capture_runtime_provenance
from llm_lifecycle_lab.tokenizer import NativeTokenizer
from llm_lifecycle_lab.training.batching import (
    DeterministicBatchStream,
    stable_stratified_subset,
)
from llm_lifecycle_lab.training.checkpoint import CheckpointManager
from llm_lifecycle_lab.training.engine import (
    EngineConfig,
    ResolvedTrainingBudget,
    TrainingEngine,
    TrainingResult,
)
from llm_lifecycle_lab.training.stages import GRPOObjective, GRPOSettings


@dataclass(frozen=True, slots=True)
class GRPORun:
    artifacts: RunArtifacts
    result: TrainingResult
    parameter_count: int
    budget: ResolvedTrainingBudget


def run_native_grpo(
    config: RunConfig,
    *,
    run_id: str | None = None,
    resume_run: str | None = None,
    resume_checkpoint: str | Path | None = None,
    workdir: str | Path = ".",
    metric_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> GRPORun:
    if config.stage is not Stage.GRPO or config.model_route is not ModelRoute.NATIVE:
        raise ConfigError("GRPO requires stage=grpo and model_route=native")
    if run_id and resume_run:
        raise ConfigError("run_id and resume_run are mutually exclusive")
    if resume_checkpoint and not resume_run:
        raise ConfigError("resume_checkpoint requires resume_run")
    root = Path(workdir).resolve()

    def required(mapping: Mapping[str, Any], name: str) -> Path:
        if not mapping.get(name):
            raise ConfigError(f"GRPO configuration must declare {name}")
        value = Path(mapping[name])
        return value if value.is_absolute() else root / value

    parent = required(config.model, "init_checkpoint")
    tokenizer_dir = required(config.model, "tokenizer")
    manifest_path = required(config.data, "manifest")
    suite_path = required(config.data, "evaluation_suite")
    metadata = CheckpointMetadata.from_dict(
        read_json(parent / "checkpoint_metadata.json")
    )
    if (
        metadata.stage not in {Stage.SFT, Stage.DPO}
        or metadata.model_route is not ModelRoute.NATIVE
    ):
        raise ConfigError(
            "Native GRPO must initialize from a Native SFT or DPO checkpoint"
        )
    tokenizer = NativeTokenizer.from_directory(tokenizer_dir)
    if metadata.tokenizer_sha256 != tokenizer.manifest.content_sha256:
        raise ConfigError("parent checkpoint and GRPO tokenizer hashes differ")
    model_config = load_native_model_config(parent / "model/config.json")
    if (
        model_config.model_id != config.model["model_id"]
        or model_config.vocab_size != tokenizer.vocab_size
    ):
        raise ConfigError("GRPO model/tokenizer does not match its parent checkpoint")
    if model_config.attention_dropout != 0:
        raise ConfigError("GRPO requires attention_dropout=0 for deterministic scoring")
    training = dict(config.training)
    settings = GRPOSettings(
        group_size=int(training.pop("group_size", 4)),
        max_new_tokens=int(training.pop("max_new_tokens", 16)),
        temperature=float(training.pop("temperature", 1.0)),
        top_p=float(training.pop("top_p", 1.0)),
        kl_beta=float(training.pop("kl_beta", 0.04)),
        advantage_epsilon=float(training.pop("advantage_epsilon", 1e-4)),
    )
    engine_config = EngineConfig.from_dict(training)
    if engine_config.sequence_length > model_config.max_sequence_length:
        raise ConfigError("GRPO sequence_length exceeds model context")
    if engine_config.eval_batches * engine_config.micro_batch_size < 2:
        raise ConfigError("GRPO dev evaluation budget must cover both languages")
    splits, summary = load_grpo_splits(
        manifest_path,
        tokenizer=tokenizer,
        sequence_length=engine_config.sequence_length,
        max_new_tokens=settings.max_new_tokens,
        evaluation_suite=suite_path,
    )
    initialization = {
        "mode": "on-policy-group-relative-policy-gradient",
        "parent_checkpoint": metadata.to_dict(),
        "parent_weights_sha256": sha256_file(parent / "model/model.pt"),
        "parent_model_config_sha256": sha256_file(parent / "model/config.json"),
        "reference": "frozen-parent",
        "data_manifest_sha256": sha256_file(manifest_path),
        "evaluation_suite_sha256": summary["suite_sha256"],
        "settings": asdict(settings),
        "optimizer_inherited": False,
        "rollout_seed": "sha256(example_id,group_index)-v1",
    }
    policy = NativeTransformer(model_config)
    policy.load(parent / "model")
    reference = NativeTransformer(model_config)
    reference.load(parent / "model")
    objective = GRPOObjective(
        tokenizer=tokenizer, reference_model=reference, settings=settings
    )
    bundle = ModelBundle(
        model=policy,
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
        metadata=ModelMetadata(
            model_route=ModelRoute.NATIVE,
            provider="native",
            model_id=model_config.model_id,
            architecture="dense-decoder",
            tokenizer_revision=tokenizer.manifest.revision,
            chat_template_version=tokenizer.chat_template_version,
            parameter_count=policy.parameter_count,
            capabilities=("forward", "generate", "kv-cache", "full-training"),
        ),
    )
    # This is a conservative upper bound used only for scheduling and progress.
    estimated_tokens = (
        len(splits["train"]) * settings.group_size * settings.max_new_tokens
    )
    engine_config, budget = engine_config.resolve_budget(
        examples_per_epoch=len(splits["train"]),
        supervised_tokens_per_epoch=estimated_tokens,
    )
    stream = DeterministicBatchStream(
        splits["train"],
        batch_size=engine_config.micro_batch_size,
        seed=config.seed,
        collate_fn=collate_grpo,
    )
    ordered = stable_stratified_subset(
        splits["dev"],
        limit=engine_config.eval_batches * engine_config.micro_batch_size,
        strata=("en", "zh"),
        stratum_field="language",
        identity_field="example_id",
    )
    evaluation = [
        collate_grpo(ordered[start : start + engine_config.micro_batch_size])
        for start in range(0, len(ordered), engine_config.micro_batch_size)
    ]
    output_root = Path(config.output_dir)
    store = ArtifactStore(
        output_root if output_root.is_absolute() else root / output_root
    )
    if resume_run:
        artifacts = store.open_run(config, run_id=resume_run)
        if read_json(artifacts.path / "initialization.json") != initialization:
            raise ArtifactError("GRPO resume inputs/reference/reward settings changed")
        checkpoint = (
            resume_checkpoint
            or read_json(artifacts.path / "latest_checkpoint.json")["path"]
        )
    else:
        artifacts = store.create_run(config, run_id=run_id)
        checkpoint = None
        artifacts.write_json("initialization.json", initialization)
        artifacts.write_json("grpo_data_summary.json", summary)
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
        stage=Stage.GRPO,
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

    def record(metrics: Mapping[str, Any]) -> None:
        artifacts.write_json(
            "latest_rollouts.json",
            {
                "schema_version": "1.0",
                "event": metrics.get("event", "train"),
                "step": metrics["step"],
                "rollouts": objective.last_rollouts,
            },
        )
        if metric_callback is not None:
            metric_callback(metrics)

    result = engine.train(
        bundle=bundle,
        objective=objective,
        train_stream=stream,
        evaluation_batches=evaluation,
        resume_from=checkpoint,
        metric_callback=record,
    )
    return GRPORun(artifacts, result, policy.parameter_count, budget)
