"""Native DPO initialized from one SFT checkpoint with a frozen reference."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab.artifacts import ArtifactStore, RunArtifacts
from llm_lifecycle_lab.config import config_sha256
from llm_lifecycle_lab.contracts import (
    CheckpointMetadata,
    ModelMetadata,
    ModelRoute,
    RunConfig,
    Stage,
)
from llm_lifecycle_lab.data.dpo import collate_dpo, load_dpo_splits
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.evaluation.suite import digest, read_json
from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError, ContractError
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
from llm_lifecycle_lab.training.stages import DPOObjective, response_logps


@dataclass(frozen=True, slots=True)
class DPORun:
    artifacts: RunArtifacts
    result: TrainingResult
    parameter_count: int
    budget: ResolvedTrainingBudget


@torch.inference_mode()
def compute_reference_scores(
    model,
    splits: dict[str, list[dict[str, Any]]],
    *,
    collate_fn,
) -> list[dict[str, Any]]:
    model.set_training(False)
    scores = []
    for split_name, examples in splits.items():
        for example in examples:
            batch = collate_fn([example])
            values = {}
            for choice in ("chosen", "rejected"):
                logps, _ = response_logps(
                    model,
                    input_ids=batch[f"{choice}_input_ids"],
                    labels=batch[f"{choice}_labels"],
                    attention_mask=batch[f"{choice}_attention_mask"],
                )
                value = float(logps[0])
                if not math.isfinite(value):
                    raise ContractError(
                        "reference model produced non-finite log-probability"
                    )
                values[choice] = value
            scores.append(
                {
                    "split": split_name,
                    "pair_id_sha256": example["pair_id_sha256"],
                    "chosen_logps": values["chosen"],
                    "rejected_logps": values["rejected"],
                }
            )
    return scores


def attach_reference_scores(
    splits: dict[str, list[dict[str, Any]]], scores: list[dict[str, Any]]
) -> None:
    index = {}
    for score in scores:
        key = (score["split"], score["pair_id_sha256"])
        if key in index:
            raise ArtifactError("duplicate frozen DPO reference score")
        if not all(
            math.isfinite(float(score[name]))
            for name in ("chosen_logps", "rejected_logps")
        ):
            raise ArtifactError("non-finite frozen DPO reference score")
        index[key] = score
    expected = {
        (split_name, example["pair_id_sha256"])
        for split_name, examples in splits.items()
        for example in examples
    }
    if set(index) != expected:
        raise ArtifactError("frozen DPO reference scores do not match preference data")
    for split_name, examples in splits.items():
        for example in examples:
            score = index[(split_name, example["pair_id_sha256"])]
            example["reference_chosen_logps"] = score["chosen_logps"]
            example["reference_rejected_logps"] = score["rejected_logps"]


def run_native_dpo(
    config: RunConfig,
    *,
    run_id: str | None = None,
    resume_run: str | None = None,
    resume_checkpoint: str | Path | None = None,
    workdir: str | Path = ".",
    metric_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> DPORun:
    if config.stage is not Stage.DPO or config.model_route is not ModelRoute.NATIVE:
        raise ConfigError("DPO requires stage=dpo and model_route=native")
    if run_id and resume_run:
        raise ConfigError("run_id and resume_run are mutually exclusive")
    if resume_checkpoint and not resume_run:
        raise ConfigError("resume_checkpoint requires resume_run")
    root = Path(workdir).resolve()

    def required(mapping: Mapping[str, Any], name: str) -> Path:
        if not mapping.get(name):
            raise ConfigError(f"DPO configuration must declare {name}")
        value = Path(mapping[name])
        return value if value.is_absolute() else root / value

    parent = required(config.model, "init_checkpoint")
    tokenizer_dir = required(config.model, "tokenizer")
    manifest_path = required(config.data, "manifest")
    suite_path = required(config.data, "evaluation_suite")
    metadata = CheckpointMetadata.from_dict(
        read_json(parent / "checkpoint_metadata.json")
    )
    if metadata.stage is not Stage.SFT or metadata.model_route is not ModelRoute.NATIVE:
        raise ConfigError("Native DPO must initialize from a Native SFT checkpoint")
    tokenizer = NativeTokenizer.from_directory(tokenizer_dir)
    if metadata.tokenizer_sha256 != tokenizer.manifest.content_sha256:
        raise ConfigError("SFT checkpoint and DPO tokenizer hashes differ")
    model_config = load_native_model_config(parent / "model/config.json")
    if (
        model_config.model_id != config.model["model_id"]
        or model_config.vocab_size != tokenizer.vocab_size
    ):
        raise ConfigError("DPO model/tokenizer does not match its SFT checkpoint")
    training = dict(config.training)
    beta = float(training.pop("beta", 0.1))
    nll_coefficient = float(training.pop("nll_coefficient", 0.0))
    if not math.isfinite(nll_coefficient) or nll_coefficient < 0:
        raise ConfigError("training.nll_coefficient must be finite and non-negative")
    engine_config = EngineConfig.from_dict(training)
    if engine_config.sequence_length > model_config.max_sequence_length:
        raise ConfigError("DPO sequence_length exceeds model context")
    if engine_config.eval_batches * engine_config.micro_batch_size < 2:
        raise ConfigError("DPO dev evaluation budget must cover both languages")
    splits, summary = load_dpo_splits(
        manifest_path,
        tokenizer=tokenizer,
        sequence_length=engine_config.sequence_length,
        evaluation_suite=suite_path,
    )
    collate = partial(collate_dpo, pad_token_id=tokenizer.pad_token_id)
    store = ArtifactStore(
        Path(config.output_dir)
        if Path(config.output_dir).is_absolute()
        else root / config.output_dir
    )
    if resume_run:
        artifacts = store.open_run(config, run_id=resume_run)
        frozen_scores = read_json(artifacts.path / "reference_scores.json")["scores"]
    else:
        reference = NativeTransformer(model_config)
        reference.load(parent / "model")
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        frozen_scores = compute_reference_scores(reference, splits, collate_fn=collate)
        artifacts = store.create_run(config, run_id=run_id)
    scores_sha256 = digest(frozen_scores)
    initialization = {
        "mode": "weights-only-new-stage",
        "parent_checkpoint": metadata.to_dict(),
        "parent_weights_sha256": sha256_file(parent / "model/model.pt"),
        "parent_model_config_sha256": sha256_file(parent / "model/config.json"),
        "reference": "frozen-parent-sft",
        "reference_scores_sha256": scores_sha256,
        "data_manifest_sha256": sha256_file(manifest_path),
        "evaluation_suite_sha256": summary["suite_sha256"],
        "beta": beta,
        "nll_coefficient": nll_coefficient,
        "optimizer_inherited": False,
    }
    attach_reference_scores(splits, frozen_scores)
    if resume_run:
        if read_json(artifacts.path / "initialization.json") != initialization:
            raise ArtifactError("DPO resume inputs/reference/objective changed")
        checkpoint = (
            resume_checkpoint
            or read_json(artifacts.path / "latest_checkpoint.json")["path"]
        )
    else:
        checkpoint = None
        artifacts.write_json("initialization.json", initialization)
        artifacts.write_json(
            "reference_scores.json",
            {"schema_version": "1.0", "scores": frozen_scores},
        )
        artifacts.write_json("dpo_data_summary.json", summary)
        artifacts.write_json("tokenizer/tokenizer_manifest.json", tokenizer.manifest)
        artifacts.write_text(
            "tokenizer/tokenizer.json",
            (tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"),
        )
    policy = NativeTransformer(model_config)
    policy.load(parent / "model")
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
    engine_config, budget = engine_config.resolve_budget(
        examples_per_epoch=len(splits["train"]),
        supervised_tokens_per_epoch=summary["splits"]["train"]["response_tokens"],
    )
    stream = DeterministicBatchStream(
        splits["train"],
        batch_size=engine_config.micro_batch_size,
        seed=config.seed,
        collate_fn=collate,
    )
    ordered = stable_stratified_subset(
        splits["dev"],
        limit=engine_config.eval_batches * engine_config.micro_batch_size,
        strata=("en", "zh"),
        stratum_field="language",
        identity_field="pair_id_sha256",
    )
    evaluation = [
        collate(ordered[start : start + engine_config.micro_batch_size])
        for start in range(0, len(ordered), engine_config.micro_batch_size)
    ]
    if not resume_run:
        artifacts.write_json("training_budget.json", budget.to_dict())
        artifacts.write_json("model_config.json", asdict(model_config))
    manager = CheckpointManager(
        artifacts,
        model_route=ModelRoute.NATIVE,
        stage=Stage.DPO,
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
        objective=DPOObjective(beta, nll_coefficient),
        train_stream=stream,
        evaluation_batches=evaluation,
        resume_from=checkpoint,
        metric_callback=metric_callback,
    )
    return DPORun(artifacts, result, policy.parameter_count, budget)
