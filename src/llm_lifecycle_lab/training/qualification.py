"""Read-only rollout qualification for GRPO parents."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab.config import config_sha256
from llm_lifecycle_lab.contracts import CheckpointMetadata, ModelRoute, RunConfig, Stage
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.grpo import (
    collate_grpo,
    evaluate_verifier_response,
    load_grpo_splits,
)
from llm_lifecycle_lab.evaluation.suite import digest, read_json
from llm_lifecycle_lab.exceptions import ConfigError, ContractError
from llm_lifecycle_lab.model.native import NativeTransformer, load_native_model_config
from llm_lifecycle_lab.tokenizer import NativeTokenizer
from llm_lifecycle_lab.training.engine import (
    EngineConfig,
    _autocast_context,
    resolve_device,
)
from llm_lifecycle_lab.training.stages import GRPOObjective, GRPOSettings


def qualify_native_grpo(
    config: RunConfig,
    *,
    source_groups: int = 64,
    minimum_mixed_group_fraction: float = 0.25,
    workdir: str | Path = ".",
) -> dict[str, Any]:
    """Sample fixed train groups without optimizer construction or updates."""

    if config.stage is not Stage.GRPO or config.model_route is not ModelRoute.NATIVE:
        raise ConfigError("GRPO qualification requires a Native GRPO config")
    if source_groups <= 0:
        raise ConfigError("source_groups must be positive")
    if (
        not math.isfinite(minimum_mixed_group_fraction)
        or not 0 < minimum_mixed_group_fraction <= 1
    ):
        raise ConfigError("minimum mixed-group fraction must be in (0, 1]")
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
            "GRPO qualification requires a Native SFT or DPO checkpoint"
        )
    tokenizer = NativeTokenizer.from_directory(tokenizer_dir)
    if metadata.tokenizer_sha256 != tokenizer.manifest.content_sha256:
        raise ConfigError("parent checkpoint and GRPO tokenizer hashes differ")
    model_config = load_native_model_config(parent / "model/config.json")
    if (
        model_config.model_id != config.model["model_id"]
        or model_config.vocab_size != tokenizer.vocab_size
    ):
        raise ConfigError("GRPO model/tokenizer does not match its parent")
    if model_config.attention_dropout != 0:
        raise ConfigError("GRPO qualification requires attention_dropout=0")

    training = dict(config.training)
    settings = GRPOSettings(
        group_size=int(training.pop("group_size", 4)),
        max_new_tokens=int(training.pop("max_new_tokens", 16)),
        temperature=float(training.pop("temperature", 1.0)),
        top_p=float(training.pop("top_p", 1.0)),
        kl_beta=float(training.pop("kl_beta", 0.04)),
        advantage_epsilon=float(training.pop("advantage_epsilon", 1e-4)),
    )
    engine = EngineConfig.from_dict(training)
    splits, summary = load_grpo_splits(
        manifest_path,
        tokenizer=tokenizer,
        sequence_length=engine.sequence_length,
        max_new_tokens=settings.max_new_tokens,
        evaluation_suite=suite_path,
        split_names=("train",),
    )
    selected, selected_group_ids = _qualification_subset(
        splits["train"],
        source_groups=source_groups,
    )

    policy = NativeTransformer(model_config)
    policy.load(parent / "model")
    reference = NativeTransformer(model_config)
    reference.load(parent / "model")
    objective = GRPOObjective(
        tokenizer=tokenizer,
        reference_model=reference,
        settings=settings,
    )
    device = resolve_device(engine.device)
    policy.to_device(device)
    policy.set_training(False)
    rollouts = []
    with torch.inference_mode(), _autocast_context(
        device,
        engine.torch_dtype,
    ):
        for example in selected:
            _, _, _, records = objective.rollouts(
                policy,
                collate_grpo([example]),
            )
            for record in records:
                reward, parseable = evaluate_verifier_response(
                    record["output"],
                    example["answer"],
                    example["verifier"],
                )
                if reward != record["reward"]:
                    raise ContractError("GRPO qualification reward drifted")
                rollouts.append(
                    {
                        **record,
                        "source_id": example["source_id"],
                        "verifier": example["verifier"],
                        "parseable": parseable,
                    }
                )

    prompt_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        prompt_groups[str(rollout["example_id"])].append(rollout)
    expected_prompts = source_groups * 2
    if len(prompt_groups) != expected_prompts or any(
        len(records) != settings.group_size for records in prompt_groups.values()
    ):
        raise ContractError("GRPO qualification rollout cardinality changed")
    mixed = {
        example_id
        for example_id, records in prompt_groups.items()
        if len({float(record["reward"]) for record in records}) > 1
    }
    successful = sum(float(record["reward"]) > 0 for record in rollouts)
    parse_failures = sum(not bool(record["parseable"]) for record in rollouts)
    mixed_by_language = {
        language: sum(
            example_id in mixed and records[0]["language"] == language
            for example_id, records in prompt_groups.items()
        )
        for language in ("en", "zh")
    }
    mixed_fraction = len(mixed) / len(prompt_groups)
    success_rate = successful / len(rollouts)
    checks = (
        _check("non-degenerate-success-rate", 0 < success_rate < 1, success_rate),
        _check(
            "mixed-reward-group-fraction",
            mixed_fraction >= minimum_mixed_group_fraction,
            mixed_fraction,
        ),
        _check(
            "mixed-reward-groups-en",
            mixed_by_language["en"] > 0,
            mixed_by_language["en"],
        ),
        _check(
            "mixed-reward-groups-zh",
            mixed_by_language["zh"] > 0,
            mixed_by_language["zh"],
        ),
        _check("verifier-parse-failures", parse_failures == 0, parse_failures),
    )
    return {
        "schema_version": "1.0",
        "ok": all(check["ok"] for check in checks),
        "config_sha256": config_sha256(config),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_metadata": metadata.to_dict(),
        "parent_weights_sha256": sha256_file(parent / "model/model.pt"),
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": sha256_file(manifest_path),
        "evaluation_suite_sha256": summary["suite_sha256"],
        "qualification_protocol_sha256": digest(
            {
                "selection": "sha256(source_id)-ascending-v1",
                "source_groups": source_groups,
                "languages_per_source_group": 2,
                "settings": asdict(settings),
                "minimum_mixed_group_fraction": minimum_mixed_group_fraction,
            }
        ),
        "settings": asdict(settings),
        "source_groups": source_groups,
        "selected_source_ids": selected_group_ids,
        "prompt_groups": len(prompt_groups),
        "rollouts": len(rollouts),
        "successful_rollouts": successful,
        "success_rate": success_rate,
        "mixed_reward_groups": len(mixed),
        "mixed_reward_group_fraction": mixed_fraction,
        "mixed_reward_groups_by_language": mixed_by_language,
        "verifier_parse_failures": parse_failures,
        "checks": checks,
        "rollout_records": rollouts,
    }


def _qualification_subset(
    examples: list[dict[str, Any]],
    *,
    source_groups: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        source_id = example.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ContractError("GRPO qualification example has no source_id")
        grouped[source_id].append(example)
    eligible = {}
    for source_id, records in grouped.items():
        by_language = {str(record["language"]): record for record in records}
        if len(records) == 2 and set(by_language) == {"en", "zh"}:
            eligible[source_id] = by_language
    if len(eligible) < source_groups:
        raise ContractError(
            f"GRPO qualification needs {source_groups} bilingual source groups; "
            f"found {len(eligible)}"
        )
    selected_ids = sorted(
        eligible,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).digest(),
    )[:source_groups]
    selected = [
        eligible[source_id][language]
        for source_id in selected_ids
        for language in ("en", "zh")
    ]
    return selected, selected_ids


def _check(name: str, ok: bool, actual: int | float) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "actual": actual}
