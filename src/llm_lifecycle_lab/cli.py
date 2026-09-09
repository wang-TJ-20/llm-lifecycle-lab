"""Command-line entry point for reproducible lifecycle operations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_lifecycle_lab import __version__
from llm_lifecycle_lab.artifacts import ArtifactStore
from llm_lifecycle_lab.config import config_sha256, load_run_config
from llm_lifecycle_lab.contracts import ModelRoute, RecordKind, Stage
from llm_lifecycle_lab.data import (
    SplitRatios,
    available_public_mixture_recipes,
    available_public_recipes,
    load_public_mixture_manifest,
    load_public_source_manifest,
    materialize_public_dataset,
    materialize_public_mixture,
    prepare_dataset,
    validate_jsonl,
)
from llm_lifecycle_lab.data.schemas import format_validation_failure
from llm_lifecycle_lab.doctor import available_profiles, run_doctor
from llm_lifecycle_lab.doctor.result import CheckStatus, DoctorReport
from llm_lifecycle_lab.exceptions import LLMLabError


def build_parser(
    *,
    command: tuple[str, ...] = (),
    prog: str = "llmlab",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run reproducible LLM lifecycle experiments.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="check environment, hardware, config, and prepared data",
    )
    doctor_parser.add_argument("--profile", choices=available_profiles())
    doctor_parser.add_argument("--config", type=Path)
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")
    doctor_parser.set_defaults(handler=_doctor_command)

    config_parser = subparsers.add_parser(
        "config",
        help="validate experiment configuration",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        required=True,
    )
    config_validate = config_subparsers.add_parser("validate")
    config_validate.add_argument("path", type=Path)
    config_validate.add_argument("--json", action="store_true", dest="as_json")
    config_validate.set_defaults(handler=_config_validate_command)

    data_parser = subparsers.add_parser(
        "data",
        help="validate and prepare JSONL datasets",
    )
    data_subparsers = data_parser.add_subparsers(
        dest="data_command",
        required=True,
    )
    data_validate = data_subparsers.add_parser("validate")
    data_validate.add_argument("--input", type=Path, required=True)
    data_validate.add_argument(
        "--kind",
        choices=[kind.value for kind in RecordKind],
        required=True,
    )
    data_validate.add_argument("--json", action="store_true", dest="as_json")
    data_validate.set_defaults(handler=_data_validate_command)

    data_recipes = data_subparsers.add_parser(
        "recipes",
        help="list built-in public dataset recipes",
    )
    data_recipes.add_argument("--json", action="store_true", dest="as_json")
    data_recipes.set_defaults(handler=_data_recipes_command)

    data_fetch = data_subparsers.add_parser(
        "fetch",
        help="materialize a pinned public dataset recipe as canonical JSONL",
    )
    data_fetch.add_argument("--recipe", required=True)
    data_fetch.add_argument("--output", type=Path, required=True)
    data_fetch.add_argument("--accept-license", required=True)
    data_fetch.set_defaults(handler=_data_fetch_command)

    data_mix = data_subparsers.add_parser(
        "mix",
        help="combine pinned public sources using a versioned mixture recipe",
    )
    data_mix.add_argument("--mixture", required=True)
    data_mix.add_argument("--input", type=Path, action="append", required=True)
    data_mix.add_argument("--output", type=Path, required=True)
    data_mix.set_defaults(handler=_data_mix_command)

    data_prepare = data_subparsers.add_parser("prepare")
    data_prepare.add_argument("--input", type=Path, required=True)
    data_prepare.add_argument("--output", type=Path, required=True)
    data_prepare.add_argument("--dataset-id", required=True)
    data_prepare.add_argument(
        "--kind",
        choices=[kind.value for kind in RecordKind],
        required=True,
    )
    data_prepare.add_argument("--license", dest="license_name", required=True)
    data_prepare.add_argument("--seed", type=int, default=42)
    data_prepare.add_argument("--group-by", default="id")
    data_prepare.add_argument("--train-ratio", type=float, default=0.8)
    data_prepare.add_argument("--dev-ratio", type=float, default=0.1)
    data_prepare.add_argument("--test-ratio", type=float, default=0.1)
    data_prepare.set_defaults(handler=_data_prepare_command)

    data_pack = data_subparsers.add_parser(
        "pack",
        help="materialize immutable memory-mapped pretraining token arrays",
    )
    data_pack.add_argument("--manifest", type=Path, required=True)
    data_pack.add_argument("--tokenizer", type=Path, required=True)
    data_pack.add_argument("--output", type=Path, required=True)
    data_pack.add_argument("--sequence-length", type=int, required=True)
    data_pack.set_defaults(handler=_data_pack_command)

    tokenizer_parser = subparsers.add_parser(
        "tokenizer",
        help="train and inspect native tokenizers",
    )
    tokenizer_subparsers = tokenizer_parser.add_subparsers(
        dest="tokenizer_command",
        required=True,
    )
    tokenizer_train = tokenizer_subparsers.add_parser("train")
    tokenizer_train.add_argument("--manifest", type=Path, required=True)
    tokenizer_train.add_argument("--output", type=Path, required=True)
    tokenizer_train.add_argument("--tokenizer-id", required=True)
    tokenizer_train.add_argument("--vocab-size", type=int, default=16_384)
    tokenizer_train.add_argument("--min-frequency", type=int, default=2)
    tokenizer_train.set_defaults(handler=_tokenizer_train_command)

    tokenizer_inspect = tokenizer_subparsers.add_parser("inspect")
    tokenizer_inspect.add_argument("path", type=Path)
    tokenizer_inspect.add_argument("--text", required=True)
    tokenizer_inspect.set_defaults(handler=_tokenizer_inspect_command)

    model_parser = subparsers.add_parser(
        "model",
        help="inspect native model configurations",
    )
    model_subparsers = model_parser.add_subparsers(
        dest="model_command",
        required=True,
    )
    model_inspect = model_subparsers.add_parser("inspect")
    model_inspect.add_argument("--config", type=Path, required=True)
    model_inspect.add_argument(
        "--compare-vocab-size",
        type=int,
        action="append",
        default=[],
        help="compare model parameter budgets for another vocabulary size",
    )
    model_inspect.add_argument("--json", action="store_true", dest="as_json")
    model_inspect.set_defaults(handler=_model_inspect_command)

    train_parser = subparsers.add_parser(
        "train",
        help="run a lifecycle training stage",
    )
    train_subparsers = train_parser.add_subparsers(
        dest="train_command",
        required=True,
    )
    train_pretrain = train_subparsers.add_parser("pretrain")
    train_pretrain.add_argument("--config", type=Path, required=True)
    train_pretrain.add_argument("--run-id")
    train_pretrain.add_argument("--resume-run")
    train_pretrain.add_argument("--resume-checkpoint", type=Path)
    train_pretrain.set_defaults(handler=_train_pretrain_command)

    eval_parser = subparsers.add_parser(
        "eval",
        help="evaluate a lifecycle checkpoint",
    )
    eval_subparsers = eval_parser.add_subparsers(
        dest="eval_command",
        required=True,
    )
    eval_pretrain = eval_subparsers.add_parser("pretrain")
    eval_pretrain.add_argument("--config", type=Path, required=True)
    eval_pretrain.add_argument("--checkpoint", type=Path, required=True)
    eval_pretrain.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
    )
    eval_pretrain.add_argument("--json", action="store_true", dest="as_json")
    eval_pretrain.set_defaults(handler=_eval_pretrain_command)

    reference_parser = subparsers.add_parser(
        "reference",
        help="verify publishable reference run artifacts",
    )
    reference_subparsers = reference_parser.add_subparsers(
        dest="reference_command",
        required=True,
    )
    reference_verify = reference_subparsers.add_parser("verify")
    reference_verify.add_argument("--spec", type=Path, required=True)
    reference_verify.add_argument("--run", type=Path, required=True)
    reference_verify.add_argument("--json", action="store_true", dest="as_json")
    reference_verify.set_defaults(handler=_reference_verify_command)

    run_parser = subparsers.add_parser(
        "run",
        help="create and inspect run artifacts",
    )
    run_subparsers = run_parser.add_subparsers(
        dest="run_command",
        required=True,
    )
    run_create = run_subparsers.add_parser("create")
    run_create.add_argument("--config", type=Path, required=True)
    run_create.add_argument("--run-id")
    run_create.set_defaults(handler=_run_create_command)

    parsers = {
        (): parser,
        ("doctor",): doctor_parser,
        ("config",): config_parser,
        ("config", "validate"): config_validate,
        ("data",): data_parser,
        ("data", "validate"): data_validate,
        ("data", "recipes"): data_recipes,
        ("data", "fetch"): data_fetch,
        ("data", "mix"): data_mix,
        ("data", "prepare"): data_prepare,
        ("data", "pack"): data_pack,
        ("tokenizer",): tokenizer_parser,
        ("tokenizer", "train"): tokenizer_train,
        ("tokenizer", "inspect"): tokenizer_inspect,
        ("model",): model_parser,
        ("model", "inspect"): model_inspect,
        ("train",): train_parser,
        ("train", "pretrain"): train_pretrain,
        ("eval",): eval_parser,
        ("eval", "pretrain"): eval_pretrain,
        ("reference",): reference_parser,
        ("reference", "verify"): reference_verify,
        ("run",): run_parser,
        ("run", "create"): run_create,
    }
    for path, command_parser in parsers.items():
        if path[: len(command)] == command:
            command_parser.prog = " ".join((prog, *path[len(command) :]))
    return parsers[command]


def main(
    argv: Sequence[str] | None = None,
    *,
    command: tuple[str, ...] = (),
    prog: str = "llmlab",
) -> int:
    """Run the full CLI or a script's focused command in the current process."""

    parser = build_parser(command=command, prog=prog)
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _doctor_command(args: argparse.Namespace) -> int:
    config = load_run_config(args.config) if args.config else None
    report = run_doctor(
        profile=args.profile,
        config=config,
        workdir=Path.cwd(),
    )
    if args.as_json:
        print(report.to_json())
    else:
        _print_doctor_report(report)
    return report.exit_code


def _config_validate_command(args: argparse.Namespace) -> int:
    config = load_run_config(args.path)
    if config.stage is Stage.PRETRAIN and config.model_route in {
        ModelRoute.NATIVE_SMOKE,
        ModelRoute.NATIVE_LEARN,
    }:
        try:
            from llm_lifecycle_lab.training.engine import EngineConfig
        except ImportError as exc:
            raise LLMLabError(
                "training dependencies are unavailable; run "
                "`python -m pip install -e '.[training]'`"
            ) from exc
        EngineConfig.from_dict(config.training)
    result = {
        "ok": True,
        "path": str(args.path),
        "sha256": config_sha256(config),
        "config": config.to_dict(),
    }
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"PASS config {args.path}")
        print(f"  sha256: {result['sha256']}")
        print(f"  route: {config.model_route.value}")
        print(f"  profile: {config.run_profile.value}")
        print(f"  stage: {config.stage.value}")
    return 0


def _data_validate_command(args: argparse.Namespace) -> int:
    result = validate_jsonl(args.input, args.kind)
    if args.as_json:
        print(result.report.to_json())
    elif result.report.ok:
        print(
            f"PASS {args.input}: {result.report.valid_records} "
            f"{result.report.record_kind.value} records"
        )
    else:
        print(format_validation_failure(result.report), file=sys.stderr)
    return 0 if result.report.ok else 1


def _data_recipes_command(args: argparse.Namespace) -> int:
    recipes = available_public_recipes()
    mixtures = available_public_mixture_recipes()
    if args.as_json:
        values = [{"recipe_type": "source", **recipe.to_dict()} for recipe in recipes]
        values.extend(
            {"recipe_type": "mixture", **mixture.to_dict()} for mixture in mixtures
        )
        print(
            json.dumps(
                values,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for recipe in recipes:
            print(
                f"{recipe.recipe_id}: {recipe.max_records} records, "
                f"{recipe.license}, {recipe.repository}@{recipe.revision}"
            )
            print(f"  {recipe.description}")
        for mixture in mixtures:
            print(
                f"{mixture.mixture_id}: "
                f"{' + '.join(mixture.components)}, {mixture.strategy}"
            )
            print(f"  {mixture.description}")
    return 0


def _data_fetch_command(args: argparse.Namespace) -> int:
    manifest = materialize_public_dataset(
        args.recipe,
        args.output,
        accepted_license=args.accept_license,
    )
    print(f"Materialized public recipe {manifest.recipe_id}")
    print(f"  path: {args.output / manifest.source_file}")
    print(f"  records: {manifest.records}")
    print(f"  sha256: {manifest.source_sha256}")
    print(f"  revision: {manifest.revision}")
    return 0


def _data_mix_command(args: argparse.Namespace) -> int:
    manifest = materialize_public_mixture(
        args.mixture,
        args.input,
        args.output,
    )
    print(f"Materialized public mixture {manifest.mixture_id}")
    print(f"  path: {args.output / manifest.source_file}")
    print(f"  records: {manifest.records}")
    print(f"  languages: {', '.join(manifest.languages)}")
    print(f"  license: {manifest.license}")
    print(f"  sha256: {manifest.source_sha256}")
    return 0


def _data_prepare_command(args: argparse.Namespace) -> int:
    public_source = load_public_source_manifest(args.input)
    public_mixture = load_public_mixture_manifest(args.input)
    source_manifest = public_source or public_mixture
    if source_manifest is not None and (
        args.license_name.casefold() != source_manifest.license.casefold()
    ):
        raise LLMLabError(
            f"--license must match public source license {source_manifest.license}"
        )
    manifest = prepare_dataset(
        args.input,
        args.output,
        dataset_id=args.dataset_id,
        record_kind=args.kind,
        license_name=args.license_name,
        seed=args.seed,
        group_by=args.group_by,
        ratios=SplitRatios(
            train=args.train_ratio,
            dev=args.dev_ratio,
            test=args.test_ratio,
        ),
        source_metadata=(
            source_manifest.to_dict() if source_manifest is not None else None
        ),
    )
    print(f"Prepared {args.output}")
    for split in manifest.splits:
        print(
            f"  {split.name}: {split.records} records, "
            f"{split.groups} groups, sha256={split.sha256}"
        )
    print(f"  manifest: {args.output / 'data_manifest.json'}")
    return 0


def _data_pack_command(args: argparse.Namespace) -> int:
    try:
        from llm_lifecycle_lab.data.packing import (
            materialize_packed_pretraining_dataset,
        )
    except ImportError as exc:
        raise LLMLabError(
            "packing dependencies are unavailable; run "
            "`python -m pip install -e '.[training]'`"
        ) from exc

    manifest = materialize_packed_pretraining_dataset(
        args.manifest,
        args.tokenizer,
        args.output,
        sequence_length=args.sequence_length,
    )
    print(f"Packed pretraining data {manifest.dataset_id}")
    print(f"  path: {args.output / 'packed_manifest.json'}")
    print(f"  sequence_length: {manifest.sequence_length}")
    for split in manifest.splits:
        print(
            f"  {split.name}: {split.examples} examples, "
            f"{split.supervised_tokens} supervised tokens"
        )
    return 0


def _run_create_command(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    artifacts = ArtifactStore(config.output_dir).create_run(
        config,
        run_id=args.run_id,
    )
    print(f"Created run {artifacts.run_id}")
    print(f"  path: {artifacts.path}")
    return 0


def _tokenizer_train_command(args: argparse.Namespace) -> int:
    try:
        from llm_lifecycle_lab.tokenizer import train_native_tokenizer
    except ImportError as exc:
        raise LLMLabError(
            "tokenizer dependencies are unavailable; run "
            "`python -m pip install -e '.[training]'`"
        ) from exc

    manifest = train_native_tokenizer(
        args.manifest,
        args.output,
        tokenizer_id=args.tokenizer_id,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(f"Trained tokenizer {manifest.tokenizer_id}")
    print(f"  path: {args.output}")
    print(f"  vocab_size: {manifest.vocab_size}")
    print(f"  sha256: {manifest.content_sha256}")
    return 0


def _tokenizer_inspect_command(args: argparse.Namespace) -> int:
    try:
        from llm_lifecycle_lab.tokenizer import NativeTokenizer
    except ImportError as exc:
        raise LLMLabError(
            "tokenizer dependencies are unavailable; run "
            "`python -m pip install -e '.[training]'`"
        ) from exc

    tokenizer = NativeTokenizer.from_directory(args.path)
    token_ids = tokenizer.encode(args.text, add_bos=True, add_eos=True)
    print(f"token_ids: {token_ids}")
    print(f"tokens: {len(token_ids)}")
    print(f"decoded: {tokenizer.decode(token_ids)}")
    return 0


def _model_inspect_command(args: argparse.Namespace) -> int:
    try:
        from llm_lifecycle_lab.model.native import (
            NativeTransformer,
            load_native_model_config,
        )
    except ImportError as exc:
        raise LLMLabError(
            "training dependencies are unavailable; run "
            "`python -m pip install -e '.[training]'`"
        ) from exc

    config = load_native_model_config(args.config)
    model = NativeTransformer(config)
    comparison_sizes = tuple(dict.fromkeys(args.compare_vocab_size))
    vocab_budget = [
        {
            "vocab_size": vocab_size,
            "parameter_count": config.parameter_count_for_vocab_size(vocab_size),
            "token_parameter_count": (
                config.token_parameter_count_for_vocab_size(vocab_size)
            ),
            "token_parameter_share": (
                config.token_parameter_share_for_vocab_size(vocab_size)
            ),
        }
        for vocab_size in comparison_sizes
    ]
    value = {
        "model_id": config.model_id,
        "parameter_count": model.parameter_count,
        "vocab_size": config.vocab_size,
        "token_parameter_count": config.token_parameter_count_for_vocab_size(
            config.vocab_size
        ),
        "token_parameter_share": config.token_parameter_share_for_vocab_size(
            config.vocab_size
        ),
        "layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "attention_heads": config.num_attention_heads,
        "key_value_heads": config.num_key_value_heads,
        "qk_norm": config.qk_norm,
        "max_sequence_length": config.max_sequence_length,
        "vocab_budget": vocab_budget,
    }
    if args.as_json:
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(f"model_id: {config.model_id}")
        print(f"parameters: {model.parameter_count:,}")
        print(
            f"layers={config.num_hidden_layers} hidden={config.hidden_size} "
            f"heads={config.num_attention_heads} "
            f"kv_heads={config.num_key_value_heads} qk_norm={config.qk_norm}"
        )
        print(
            f"token_parameters={value['token_parameter_count']:,} "
            f"share={value['token_parameter_share']:.2%}"
        )
        for item in vocab_budget:
            print(
                f"vocab={item['vocab_size']}: "
                f"parameters={item['parameter_count']:,} "
                f"token_parameters={item['token_parameter_count']:,} "
                f"share={item['token_parameter_share']:.2%}"
            )
    return 0


def _train_pretrain_command(args: argparse.Namespace) -> int:
    try:
        from llm_lifecycle_lab.training.pretrain import run_native_pretraining
    except ImportError as exc:
        raise LLMLabError(
            "training dependencies are unavailable; run "
            "`python -m pip install -e '.[training]'`"
        ) from exc

    config = load_run_config(args.config)
    run = run_native_pretraining(
        config,
        run_id=args.run_id,
        resume_run=args.resume_run,
        resume_checkpoint=args.resume_checkpoint,
        workdir=Path.cwd(),
        metric_callback=_print_training_metric,
    )
    print(f"Completed pretraining run {run.artifacts.run_id}")
    print(f"  parameters: {run.parameter_count:,}")
    print(f"  steps: {run.result.global_step}")
    print(f"  tokens_seen: {run.result.tokens_seen}")
    print(f"  target_train_tokens: {run.result.target_train_tokens}")
    print(f"  epochs_seen: {run.result.epochs_seen:.4f}")
    print(f"  target_token_coverage: {run.result.target_token_coverage:.2%}")
    print(f"  final_loss: {run.result.final_loss:.6f}")
    print(f"  checkpoint: {run.result.final_checkpoint}")
    return 0


def _eval_pretrain_command(args: argparse.Namespace) -> int:
    try:
        from llm_lifecycle_lab.training.pretrain import (
            evaluate_native_pretraining,
        )
    except ImportError as exc:
        raise LLMLabError(
            "training dependencies are unavailable; run "
            "`python -m pip install -e '.[training]'`"
        ) from exc

    config = load_run_config(args.config)
    report = evaluate_native_pretraining(
        config,
        checkpoint=args.checkpoint,
        split=args.split,
        workdir=Path.cwd(),
    )
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"checkpoint_step={report['checkpoint_step']} split={report['split']} "
            f"loss={report['eval_loss']:.6f} "
            f"perplexity={report['eval_perplexity']:.4f}"
        )
        if "eval_bits_per_byte" in report:
            print(f"bits_per_byte={report['eval_bits_per_byte']:.6f}")
        for language in ("en", "zh"):
            loss_key = f"eval_{language}_loss"
            if loss_key not in report:
                continue
            print(
                f"{language}: loss={report[loss_key]:.6f} "
                f"perplexity={report[f'eval_{language}_perplexity']:.4f} "
                f"bits_per_byte="
                f"{report[f'eval_{language}_bits_per_byte']:.6f}"
            )
    return 0


def _reference_verify_command(args: argparse.Namespace) -> int:
    try:
        from llm_lifecycle_lab.reference import verify_reference_run
    except ImportError as exc:
        raise LLMLabError(
            "reference verification dependencies are unavailable; run "
            "`python -m pip install -e '.[training]'`"
        ) from exc

    report = verify_reference_run(
        args.spec,
        args.run,
        workdir=Path.cwd(),
    )
    if args.as_json:
        print(report.to_json())
    else:
        for check in report.checks:
            label = "PASS" if check.status is CheckStatus.PASS else "FAIL"
            print(f"{label:4} {check.name}: {check.message}")
        counts = report.counts
        print(
            f"Summary: {counts['pass']} passed, "
            f"{counts['warn']} warnings, {counts['fail']} failed"
        )
    return report.exit_code


def _print_training_metric(metric: Mapping[str, object]) -> None:
    if metric.get("event") == "training-plan":
        print(
            f"plan mode={metric['mode']} max_steps={metric['max_steps']} "
            f"target_tokens={metric['target_train_tokens']} "
            f"estimated_epochs={float(metric['estimated_epochs']):.4f}"
        )
        return
    step = metric.get("step", "?")
    if "train_loss" in metric:
        message = f"step={step} train_loss={float(metric['train_loss']):.6f}"
        if "eval_loss" in metric:
            message += f" eval_loss={float(metric['eval_loss']):.6f}"
        print(message)
        return
    if "eval_loss" in metric:
        print(
            f"{metric.get('event', 'evaluation')} step={step} "
            f"eval_loss={float(metric['eval_loss']):.6f}"
        )


def _print_doctor_report(report: DoctorReport) -> None:
    for check in report.checks:
        label = {
            CheckStatus.PASS: "PASS",
            CheckStatus.WARN: "WARN",
            CheckStatus.FAIL: "FAIL",
        }[check.status]
        print(f"{label:4} {check.name}: {check.message}")
    counts = report.counts
    print(
        f"Summary: {counts['pass']} passed, "
        f"{counts['warn']} warnings, {counts['fail']} failed"
    )
