"""Prepare and freeze A10 SFT inputs. This script never starts training."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import yaml
from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.artifacts import _atomic_write_text
from llm_lifecycle_lab.config import config_sha256, load_mapping, load_run_config
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.posttraining import (
    _download_sources,
    _read_source_rows,
    _transform_source,
    _write_jsonl,
)
from llm_lifecycle_lab.data.prepare import prepare_dataset, verify_data_manifest
from llm_lifecycle_lab.data.sft import load_sft_splits
from llm_lifecycle_lab.data.sft_public import (
    _legacy_transform_recipe,
    load_public_sft_source_manifest,
    materialize_public_sft,
    select_public_sft_records,
)
from llm_lifecycle_lab.evaluation.suite import read_json
from llm_lifecycle_lab.exceptions import ContractError, LLMLabError
from llm_lifecycle_lab.sft_data_gate import verify_sft_data_gate
from llm_lifecycle_lab.sft_initialization import verify_sft_initialization
from llm_lifecycle_lab.tokenizer import NativeTokenizer
from llm_lifecycle_lab.training.batching import DeterministicTokenQuotaBatchStream
from llm_lifecycle_lab.training.engine import EngineConfig
from llm_lifecycle_lab.training.sft import _resolve_sampling

ARCHIVE = Path("results/native-60m-a10-base-v1-s42")
PACKAGE = Path("models/a10-base-v1-initialization")
BASE_GATE = Path("configs/gates/native-60m-a10-sft-v1-base-init.yaml")
DATA_GATE = Path("configs/gates/native-60m-a10-sft-v1-data.yaml")
RUN_GATE = Path("configs/gates/native-60m-a10-sft-v1-run.yaml")
RECIPE = Path("configs/data/sft-public-a10-v1.yaml")
RAW = Path("data/raw/sft-public-a10-v1")
PREPARED = Path("data/prepared/sft-public-a10-v1")
PIPELINE = Path("configs/pipelines/native-60m-a10-sft-v1.yaml")
SUITE = Path("configs/evaluation/lifecycle-v4.yaml")
WEIGHTS_SHA256 = "0c3290d6c76bbf775aba20c1d543a062e26aafb987a7d15bde1b652e4dbbb46c"
TASKS = ("general", "short_qa", "classification", "numeric", "structured")
TRAIN_TOKEN_BUDGET = 1_000_000


def freeze_yaml(path: Path, value: dict, *, refreeze: bool = False) -> None:
    text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    if path.exists() and path.read_text(encoding="utf-8") != text and not refreeze:
        raise ContractError(f"frozen input differs; use a new version: {path}")
    if not path.exists() or refreeze:
        _atomic_write_text(path, text)


def fingerprint(path: Path) -> dict:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def restore_manifest_identity(path: Path, snapshot: Path) -> None:
    """Preserve the original freeze time/path only after content identity agrees."""
    if not snapshot.exists():
        _atomic_write_text(snapshot, path.read_text(encoding="utf-8"))
        return
    actual = read_json(path)
    expected = read_json(snapshot)
    for value in (actual, expected):
        value.pop("created_at", None)
        if isinstance(value.get("source"), dict):
            value["source"].pop("path", None)
    if actual != expected:
        raise ContractError(
            f"materialized manifest content differs from freeze: {path}"
        )
    _atomic_write_text(path, snapshot.read_text(encoding="utf-8"))


def prepare_base(root: Path) -> None:
    archive = root / ARCHIVE
    acceptance = read_json(archive / "a10-acceptance.json")
    if (
        acceptance["status"] != "passed"
        or acceptance["source"]["training_dirty"] is not False
        or acceptance["identity"]["weights_sha256"] != WEIGHTS_SHA256
    ):
        raise ContractError("expected accepted, clean-source A10 Base")
    for name, digest in read_json(archive / "checksums.json")["files"].items():
        if sha256_file(archive / name) != digest:
            raise ContractError(f"A10 archive integrity failed: {name}")
    checkpoint = read_json(archive / "final_checkpoint/checkpoint_metadata.json")
    tokenizer = read_json(archive / "tokenizer_manifest.json")
    model = read_json(archive / "model_manifest.json")
    source = acceptance["source"]
    license_id = read_json(archive / "data_snapshot.json")["license"]
    target = root / PACKAGE
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".a10-base.", dir=target.parent))
        try:
            for name, original in {
                "config.json": "final_checkpoint/model/config.json",
                "model.pt": "final_checkpoint/model/model.pt",
                "tokenizer.json": "tokenizer/tokenizer.json",
                "tokenizer_manifest.json": "tokenizer_manifest.json",
            }.items():
                shutil.copyfile(archive / original, temporary / name)
            evidence = temporary / "source_evidence"
            evidence.mkdir()
            for name in (
                "final-test.json",
                "metrics.jsonl",
                "model_manifest.json",
                "resolved_config.yaml",
                "run_manifest.json",
                "tokenizer_manifest.json",
                "training_budget.json",
                "training_result.json",
            ):
                shutil.copyfile(archive / name, evidence / name)
            shutil.copyfile(
                archive / "final_checkpoint/checkpoint_metadata.json",
                evidence / "checkpoint_metadata.json",
            )
            provenance = {
                "package": {
                    "model_id": "tiny-60m",
                    "model_route": "native",
                    "stage": "pretrain",
                    "license": {"identifier": license_id},
                },
                "source_run": {
                    "run_id": acceptance["run_id"],
                    "run_profile": "reproduce",
                    "seed": 42,
                    "status": "completed",
                    "dirty": False,
                    "git_commit": source["commit"],
                    "source_sha256": source["source_sha256"],
                    "resolved_config_sha256": checkpoint["config_sha256"],
                    "provenance_deviation": None,
                },
                "checkpoint": {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "global_step": checkpoint["step"],
                    "created_at": checkpoint["created_at"],
                    "config_sha256": checkpoint["config_sha256"],
                    "tokenizer_sha256": checkpoint["tokenizer_sha256"],
                },
                "weights": {"sha256": WEIGHTS_SHA256, "license": license_id},
                "model_config": {
                    "model_id": "tiny-60m",
                    "sha256": sha256_file(temporary / "config.json"),
                },
                "tokenizer": {
                    "tokenizer_id": tokenizer["tokenizer_id"],
                    "vocab_size": tokenizer["vocab_size"],
                    "chat_template_version": tokenizer["chat_template_version"],
                    "content_sha256": tokenizer["content_sha256"],
                    "license": license_id,
                },
            }
            _atomic_write_text(
                temporary / "provenance.json",
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            )
            _atomic_write_text(
                temporary / "LICENSE_MODEL",
                f"Local A10 Base initialization bundle.\n"
                f"Conservative source-license declaration: {license_id}\n"
                "Weights are identical to the accepted A10 Base, not a new model.\n"
                "Source attribution: SimpleStories/SimpleStories (MIT), "
                "wikimedia/wikipedia (CC-BY-SA-3.0).\n"
                "Preserve attribution and applicable share-alike obligations; "
                "the Apache-2.0 code license is not a claim about model weights.\n",
            )
            names = (
                "LICENSE_MODEL",
                "config.json",
                "model.pt",
                "provenance.json",
                "tokenizer.json",
                "tokenizer_manifest.json",
            )
            _atomic_write_text(
                temporary / "SHA256SUMS",
                "".join(f"{sha256_file(temporary / n)}  {n}\n" for n in names),
            )
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    package_names = (
        "LICENSE_MODEL",
        "config.json",
        "model.pt",
        "provenance.json",
        "tokenizer.json",
        "tokenizer_manifest.json",
    )
    evidence_names = tuple(
        sorted(p.name for p in (target / "source_evidence").iterdir() if p.is_file())
    )
    spec = {
        "schema_version": "1.0",
        "gate_id": "native-60m-a10-sft-v1-base-init",
        "paths": {
            "base_package": str(PACKAGE),
            "source_evidence": str(PACKAGE / "source_evidence"),
            "source_pipeline": "configs/pipelines/native-60m-a10-base-v1.yaml",
            "source_model_config": "configs/models/tiny-60m.yaml",
            "release_manifest": str(PACKAGE / "SHA256SUMS"),
        },
        "package_artifacts": {n: fingerprint(target / n) for n in package_names},
        "evidence_artifacts": {
            n: fingerprint(target / "source_evidence" / n) for n in evidence_names
        },
        "identity": {
            "model_route": "native",
            "model_id": "tiny-60m",
            "architecture": "dense-decoder",
            "source_stage": "pretrain",
            "source_run_id": acceptance["run_id"],
            "source_run_profile": "reproduce",
            "source_seed": 42,
            "source_git_commit": source["commit"],
            "source_sha256": source["source_sha256"],
            "source_run_config_sha256": checkpoint["config_sha256"],
            "source_pipeline_file_sha256": sha256_file(
                root / "configs/pipelines/native-60m-a10-base-v1.yaml"
            ),
            "source_model_config_sha256": model["config_sha256"],
            "weights_sha256": WEIGHTS_SHA256,
            "tokenizer_sha256": tokenizer["content_sha256"],
            "tokenizer_manifest_sha256": sha256_file(
                target / "tokenizer_manifest.json"
            ),
            "tokenizer_id": tokenizer["tokenizer_id"],
            "chat_template_version": tokenizer["chat_template_version"],
            "parameter_count": 62927616,
            "vocab_size": 16384,
            "max_sequence_length": 512,
            "checkpoint_step": 45191,
            "license": license_id,
        },
        "quality": {
            "expected_source_status": "completed",
            "expected_global_step": 45191,
            "expected_target_train_tokens": 369476240,
            "minimum_target_token_coverage": 1.0,
            "maximum_target_token_coverage": 1.01,
            "maximum_final_to_baseline_loss_ratio": 0.5,
            "maximum_final_to_best_loss_ratio": 1.02,
            "require_final_language_improvement": True,
            "expected_test_split": "test",
        },
        "policy": {
            "initialization_mode": "weights-only-new-stage",
            "inherit_optimizer_state": False,
            "initial_global_step": 0,
            "require_clean_git": True,
            "accepted_source_dirty": False,
            "require_source_deviation_record": False,
        },
    }
    freeze_yaml(root / BASE_GATE, spec)
    require_pass(verify_sft_initialization(root / BASE_GATE, workdir=root))
    print("A10 Base initialization verified; no training started.", flush=True)


def require_pass(report) -> None:
    failures = [
        f"{c.name}: {c.details}" for c in report.checks if c.status.value == "fail"
    ]
    if failures:
        raise ContractError("; ".join(failures))


def prepare_data(root: Path, accepted: list[str], *, refreeze: bool = False) -> None:
    plan = load_mapping(root / "configs/data/a10-sft-public-plan.yaml")
    licenses = {source["license"].casefold() for source in plan["sources"]}
    if {value.casefold() for value in accepted} != licenses:
        raise ContractError("confirm all plan licenses using repeated --accept-license")
    tokenizer = NativeTokenizer.from_directory(root / PACKAGE)
    if not (root / RECIPE).exists():
        raw = copy.deepcopy(plan)
        raw["output"] = {"record_kind": "sft", "records": 1, "sha256": None}
        adapted = _legacy_transform_recipe(raw)
        print("Fetching pinned public SFT inputs (preparation only)...", flush=True)
        paths, _ = _download_sources(adapted)
        candidates = []
        for source in adapted.sources:
            transformed = _transform_source(
                _read_source_rows(paths[source.source_id], source), adapted, source
            )
            from llm_lifecycle_lab.contracts import RecordKind

            candidates.extend(transformed.get(RecordKind.SFT, ()))
        records, processing = select_public_sft_records(candidates, tokenizer=tokenizer)
        with tempfile.TemporaryDirectory(prefix="a10-sft-freeze-") as directory:
            source_path = Path(directory) / "source.jsonl"
            _write_jsonl(source_path, records)
            raw["output"] = {
                "record_kind": "sft",
                "records": len(records),
                "sha256": sha256_file(source_path),
                "tokenizer_sha256": tokenizer.manifest.content_sha256,
                "max_sequence_length": 512,
            }
        freeze_yaml(root / RECIPE, raw)
        print(f"Public SFT filtering: {processing}", flush=True)
    recipe = load_mapping(root / RECIPE)
    if {key: value for key, value in recipe.items() if key != "output"} != plan:
        raise ContractError("frozen recipe differs from the source plan")
    if not (root / RAW).exists():
        materialize_public_sft(
            root / RECIPE,
            root / RAW,
            accepted_licenses=accepted,
            tokenizer_dir=root / PACKAGE,
        )
    source = root / RAW / "source.jsonl"
    source_manifest = load_public_sft_source_manifest(source)
    if (
        source_manifest is None
        or source_manifest.source_sha256 != recipe["output"]["sha256"]
        or source_manifest.records != recipe["output"]["records"]
    ):
        raise ContractError("materialized source differs from frozen recipe")
    restore_manifest_identity(
        source.parent / "sft_source_manifest.json",
        root / "configs/reference/a10-sft-source-manifest.json",
    )
    source_manifest = load_public_sft_source_manifest(source)
    if not (root / PREPARED).exists():
        prepare_dataset(
            source,
            root / PREPARED,
            dataset_id=recipe["preparation"]["dataset_id"],
            record_kind="sft",
            license_name=source_manifest.license,
            seed=42,
            group_by="source_id",
            source_metadata=source_manifest.to_dict(),
        )
    manifest_path = root / PREPARED / "data_manifest.json"
    failures = verify_data_manifest(manifest_path)
    if failures:
        raise ContractError("; ".join(failures))
    restore_manifest_identity(
        manifest_path, root / "configs/reference/a10-sft-data-manifest.json"
    )
    splits, summary = load_sft_splits(
        manifest_path,
        tokenizer=tokenizer,
        sequence_length=512,
        evaluation_suite=root / SUITE,
    )
    train = summary["splits"]["train"]
    preparation = {k: v for k, v in recipe["preparation"].items() if k != "dataset_id"}
    data_spec = {
        "schema_version": "1.0",
        "gate_id": "native-60m-a10-sft-v1-data",
        "locks": {
            "base_gate": str(BASE_GATE),
            "base_gate_sha256": sha256_file(root / BASE_GATE),
            "recipe": str(RECIPE),
            "recipe_sha256": sha256_file(root / RECIPE),
        },
        "paths": {
            "canonical_source": str(RAW / "source.jsonl"),
            "prepared_manifest": str(PREPARED / "data_manifest.json"),
            "tokenizer": str(PACKAGE),
        },
        "identity": {
            "recipe_id": recipe["recipe_id"],
            "selection_namespace": recipe["selection_namespace"],
            "dataset_id": recipe["preparation"]["dataset_id"],
            "record_kind": "sft",
            "source_records": source_manifest.records,
            "source_sha256": source_manifest.source_sha256,
            "tokenizer_id": tokenizer.manifest.tokenizer_id,
            "tokenizer_sha256": tokenizer.manifest.content_sha256,
            "tokenizer_manifest_sha256": sha256_file(
                root / PACKAGE / "tokenizer_manifest.json"
            ),
            "chat_template_version": tokenizer.chat_template_version,
        },
        "preparation": preparation,
        "quality": {
            "required_languages": ["en", "zh"],
            "max_sequence_length": 512,
            "train_examples": train["examples"],
            "train_supervised_tokens": train["supervised_tokens"],
            "train_language_examples": train["languages"],
            "train_task_examples": {
                task: train["task_families"][task]["examples"] for task in TASKS
            },
        },
        "policy": {
            "public_only": True,
            "allow_synthetic": False,
            "require_pinned_revisions": True,
            "require_pinned_file_hashes": True,
            "allowed_source_ids": [s["source_id"] for s in recipe["sources"]],
            "excluded_source_ids": ["helpsteer3"],
            "supervision": "assistant-only",
            "truncation": "forbidden",
            "duplicate_normalization": "NFKC-whitespace-collapse-v1",
            "require_clean_git": True,
        },
    }
    freeze_yaml(root / DATA_GATE, data_spec)
    require_pass(
        verify_sft_data_gate(root / DATA_GATE, workdir=root, check_inputs=True)
    )
    freeze_pipelines(root, splits, summary, refreeze=refreeze)
    _atomic_write_text(
        root / PREPARED / "a10-sft-preparation.json",
        json.dumps(
            {
                "status": "inputs-verified-not-trained",
                "processing": source_manifest.processing,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(
        "SFT inputs frozen. Commit inputs, then start smoke/pilot yourself.", flush=True
    )


def balanced_stratum_weights(strata: dict) -> dict[str, float]:
    counts = {
        language: sum(name.endswith(f".{language}") for name in strata)
        for language in ("en", "zh")
    }
    if not all(counts.values()):
        raise ContractError("both languages require non-empty sampling strata")
    return {name: 0.5 / counts[name.rsplit(".", 1)[1]] for name in sorted(strata)}


def freeze_pipelines(
    root: Path, splits: dict, summary: dict, *, refreeze: bool = False
) -> None:
    if refreeze and any((root / "runs").glob("native-60m-a10-sft-v1*")):
        raise ContractError(
            "cannot refreeze execution inputs after an A10 SFT run exists"
        )
    training = {
        "device": "cuda",
        "dtype": "bfloat16",
        "sequence_length": 512,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_train_tokens": TRAIN_TOKEN_BUDGET,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.95,
        "warmup_steps": 50,
        "min_lr_ratio": 0.1,
        "gradient_clipping": 1.0,
        "checkpoint_interval": 1000,
        "eval_interval": 500,
        "eval_batches": len(splits["dev"]),
        "log_interval": 10,
    }
    config = {
        "schema_version": "1.0",
        "model_route": "native",
        "run_profile": "reproduce",
        "stage": "sft",
        "seed": 42,
        "output_dir": "runs",
        "model": {
            "provider": "native",
            "model_id": "tiny-60m",
            "architecture": "dense-decoder",
            "base_package": str(PACKAGE),
            "tokenizer": str(PACKAGE),
        },
        "data": {
            "manifest": str(PREPARED / "data_manifest.json"),
            "base_gate": str(BASE_GATE),
            "data_gate": str(DATA_GATE),
            "evaluation_suite": str(SUITE),
            "evaluation_suite_sha256": sha256_file(root / SUITE),
            "sampling": {
                "strategy": "supervised-token-quota",
                "weights": balanced_stratum_weights(
                    summary["splits"]["train"]["sampling_strata"]
                ),
            },
        },
        "training": training,
    }
    freeze_yaml(root / PIPELINE, config, refreeze=refreeze)
    for suffix, steps in (("smoke", 8), ("pilot100", 100)):
        development = copy.deepcopy(config)
        development["run_profile"] = "smoke"
        settings = development["training"]
        settings.pop("max_train_tokens")
        settings.update(
            max_steps=steps,
            warmup_steps=2 if steps == 8 else 10,
            checkpoint_interval=4 if steps == 8 else 50,
            eval_interval=4 if steps == 8 else 50,
            eval_batches=16,
            log_interval=1,
        )
        freeze_yaml(
            root / f"configs/pipelines/native-60m-a10-sft-v1-{suffix}.yaml",
            development,
            refreeze=refreeze,
        )
    engine = EngineConfig.from_dict(training)
    sampling = _resolve_sampling(config["data"], summary["splits"]["train"])
    resolved, budget = engine.resolve_budget(
        examples_per_epoch=len(splits["train"]),
        supervised_tokens_per_epoch=sampling["estimated_tokens_per_virtual_epoch"],
    )
    stream = DeterministicTokenQuotaBatchStream(
        splits["train"],
        batch_size=resolved.micro_batch_size,
        seed=42,
        collate_fn=lambda rows: sum(row["supervised_token_count"] for row in rows),
        stratum_field=sampling["stratum_field"],
        token_count_field="supervised_token_count",
        weights=sampling["weights"],
    )
    tokens_seen = sum(
        stream.next_batch()
        for _ in range(resolved.max_steps * resolved.gradient_accumulation_steps)
    )
    if not 0.99 <= tokens_seen / TRAIN_TOKEN_BUDGET <= 1.01:
        raise ContractError(
            "predicted token coverage outside the predeclared 1% bounds"
        )
    freeze_yaml(
        root / RUN_GATE,
        {
            "schema_version": "1.0",
            "gate_id": "native-60m-a10-sft-v1-run",
            "locks": {
                "pipeline_config": str(PIPELINE),
                "pipeline_file_sha256": sha256_file(root / PIPELINE),
                "pipeline_config_sha256": config_sha256(
                    load_run_config(root / PIPELINE)
                ),
                "base_gate": str(BASE_GATE),
                "base_gate_sha256": sha256_file(root / BASE_GATE),
                "data_gate": str(DATA_GATE),
                "data_gate_sha256": sha256_file(root / DATA_GATE),
                "evaluation_suite": str(SUITE),
                "evaluation_suite_file_sha256": sha256_file(root / SUITE),
                "evaluation_suite_sha256": summary["suite_sha256"],
            },
            "identity": {
                "model_route": "native",
                "run_profile": "reproduce",
                "stage": "sft",
                "parent_run_id": "native-60m-a10-base-v1-s42",
                "parent_checkpoint_step": 45191,
                "parent_weights_sha256": WEIGHTS_SHA256,
                "parent_model_config_sha256": sha256_file(
                    root / PACKAGE / "config.json"
                ),
                "tokenizer_sha256": read_json(
                    root / PACKAGE / "tokenizer_manifest.json"
                )["content_sha256"],
                "data_manifest_sha256": sha256_file(
                    root / PREPARED / "data_manifest.json"
                ),
            },
            "training": {
                "budget_mode": "max_train_tokens",
                "global_step": resolved.max_steps,
                "target_train_tokens": TRAIN_TOKEN_BUDGET,
                "expected_tokens_seen": tokens_seen,
                "minimum_target_token_coverage": 0.99,
                "maximum_target_token_coverage": 1.01,
                "required_training_metric_keys": [
                    "train_loss",
                    "gradient_norm",
                    "eval_loss",
                    "eval_en_loss",
                    "eval_zh_loss",
                    "eval_tokens",
                    "eval_en_tokens",
                    "eval_zh_tokens",
                ],
            },
            "evaluation": {
                "dev_suite": "sft-dev",
                "dev_examples": len(splits["dev"]),
                "test_suite": "sft-test",
                "test_examples": len(splits["test"]),
                "required_metric_keys": [
                    "eval_loss",
                    "eval_perplexity",
                    "eval_tokens",
                    "eval_en_loss",
                    "eval_zh_loss",
                    "eval_en_perplexity",
                    "eval_zh_perplexity",
                    "eval_en_tokens",
                    "eval_zh_tokens",
                ],
            },
            "runtime": {
                "platform_system": "Linux",
                "accelerator_type": "cuda",
                "require_clean_source": True,
            },
        },
        refreeze=refreeze,
    )
    print(
        f"Budget without training: {asdict(budget)}, predicted tokens={tokens_seen}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument(
        "--refreeze-untrained",
        action="store_true",
        help="explicitly rebuild execution inputs, only before any A10 SFT run exists",
    )
    parser.add_argument("--accept-license", action="append", default=[])
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        prepare_base(root)
        if not args.base_only:
            prepare_data(root, args.accept_license, refreeze=args.refreeze_untrained)
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
