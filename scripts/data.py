"""准备可复现的预训练与后训练数据。

本脚本提供公开数据下载、混合、校验、切分与打包子命令。
预训练切分后需单独训练分词器并执行 ``pack``；后训练复用已有分词器。
各步骤直接调用 ``src/llm_lifecycle_lab/data`` 下对应模块的算法实现。

Prepare reproducible pretraining and post-training data.

This script provides public-data fetch, validation, preparation, and packing.
Pretraining requires tokenizer training and packing; post-training reuses one.
Each step directly calls its algorithm in ``src/llm_lifecycle_lab/data``.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.contracts import RecordKind
from llm_lifecycle_lab.data.mix import (
    available_public_mixture_recipes,
    load_public_mixture_manifest,
    materialize_public_mixture,
)
from llm_lifecycle_lab.data.posttraining import (
    available_public_posttraining_recipes,
    load_public_posttraining_source_manifest,
    materialize_public_posttraining,
)
from llm_lifecycle_lab.data.prepare import prepare_dataset
from llm_lifecycle_lab.data.public import (
    available_public_recipes,
    load_public_source_manifest,
    materialize_public_dataset,
)
from llm_lifecycle_lab.data.schemas import format_validation_failure, validate_jsonl
from llm_lifecycle_lab.data.split import SplitRatios
from llm_lifecycle_lab.exceptions import LLMLabError

_POSTTRAIN_STAGES = (RecordKind.SFT, RecordKind.DPO, RecordKind.GRPO)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/data.py",
        description="Fetch, validate, prepare, and pack lifecycle training data.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    recipes = commands.add_parser("recipes", help="list built-in data recipes")
    recipes.add_argument("--json", action="store_true", dest="as_json")

    fetch = commands.add_parser("fetch", help="download one pinned source recipe")
    fetch.add_argument("--recipe", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--accept-license", required=True)

    posttrain_recipes = commands.add_parser(
        "posttrain-recipes",
        help="list built-in public post-training recipes",
    )
    posttrain_recipes.add_argument("--json", action="store_true", dest="as_json")

    fetch_posttrain = commands.add_parser(
        "fetch-posttrain",
        help="download pinned public SFT, DPO, and GRPO sources",
    )
    fetch_posttrain.add_argument("--recipe", required=True)
    fetch_posttrain.add_argument("--output", type=Path, required=True)
    fetch_posttrain.add_argument(
        "--accept-license",
        action="append",
        required=True,
        help="repeat once for each license required by the recipe",
    )

    mix = commands.add_parser("mix", help="combine sources using a mixture recipe")
    mix.add_argument("--mixture", required=True)
    mix.add_argument("--input", type=Path, action="append", required=True)
    mix.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate", help="validate a JSONL dataset")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument(
        "--kind",
        choices=[kind.value for kind in RecordKind],
        required=True,
    )
    validate.add_argument("--json", action="store_true", dest="as_json")

    prepare = commands.add_parser(
        "prepare",
        help="split validated JSONL into train, dev, and test sets",
    )
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--dataset-id", required=True)
    prepare.add_argument(
        "--kind",
        choices=[kind.value for kind in RecordKind],
        required=True,
    )
    prepare.add_argument("--license", dest="license_name", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--group-by", default="id")
    prepare.add_argument("--train-ratio", type=float, default=0.8)
    prepare.add_argument("--dev-ratio", type=float, default=0.1)
    prepare.add_argument("--test-ratio", type=float, default=0.1)

    check_posttrain = commands.add_parser(
        "check-posttrain",
        help="load all prepared post-training splits and verify stage isolation",
    )
    check_posttrain.add_argument("--sft-manifest", type=Path, required=True)
    check_posttrain.add_argument("--dpo-manifest", type=Path, required=True)
    check_posttrain.add_argument("--grpo-manifest", type=Path, required=True)
    check_posttrain.add_argument("--tokenizer", type=Path, required=True)
    check_posttrain.add_argument("--evaluation-suite", type=Path, required=True)
    check_posttrain.add_argument("--sequence-length", type=int, default=512)
    check_posttrain.add_argument("--max-new-tokens", type=int, default=16)
    check_posttrain.add_argument("--json", action="store_true", dest="as_json")

    pack = commands.add_parser(
        "pack",
        help="convert tokenized documents into fixed-length training arrays",
    )
    pack.add_argument("--manifest", type=Path, required=True)
    pack.add_argument("--tokenizer", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--sequence-length", type=int, required=True)
    return parser


def list_recipes(args: argparse.Namespace) -> int:
    recipes = available_public_recipes()
    mixtures = available_public_mixture_recipes()
    if args.as_json:
        values = [{"recipe_type": "source", **recipe.to_dict()} for recipe in recipes]
        values.extend(
            {"recipe_type": "mixture", **mixture.to_dict()} for mixture in mixtures
        )
        print(json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

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


def fetch_source(args: argparse.Namespace) -> int:
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


def list_posttraining_recipes(args: argparse.Namespace) -> int:
    recipes = available_public_posttraining_recipes()
    if args.as_json:
        print(
            json.dumps(
                [recipe.to_dict() for recipe in recipes],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    for recipe in recipes:
        sources = ", ".join(source.repository for source in recipe.sources)
        print(f"{recipe.recipe_id}: {sources}")
        print(f"  {recipe.description}")
    return 0


def fetch_posttraining(args: argparse.Namespace) -> int:
    manifest = materialize_public_posttraining(
        args.recipe,
        args.output,
        accepted_licenses=args.accept_license,
    )
    print(f"Materialized public post-training recipe {manifest.recipe_id}")
    for stage in manifest.stages:
        print(
            f"  {stage['record_kind']}: {stage['records']} records, "
            f"sha256={stage['source_sha256']}"
        )
    print(f"  manifest: {args.output / 'bundle_manifest.json'}")
    return 0


def mix_sources(args: argparse.Namespace) -> int:
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


def validate_source(args: argparse.Namespace) -> int:
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


def prepare_splits(args: argparse.Namespace) -> int:
    source_manifest = load_public_posttraining_source_manifest(args.input)
    if source_manifest is None:
        source_manifest = load_public_source_manifest(args.input)
    if source_manifest is None:
        source_manifest = load_public_mixture_manifest(args.input)
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


def check_posttraining_data(args: argparse.Namespace) -> int:
    from itertools import combinations

    from llm_lifecycle_lab.data.dpo import load_dpo_splits
    from llm_lifecycle_lab.data.grpo import load_grpo_splits
    from llm_lifecycle_lab.data.prepare import load_data_manifest
    from llm_lifecycle_lab.data.sft import load_sft_splits
    from llm_lifecycle_lab.exceptions import DataValidationError
    from llm_lifecycle_lab.tokenizer import NativeTokenizer

    tokenizer = NativeTokenizer.from_directory(args.tokenizer)
    _, sft_summary = load_sft_splits(
        args.sft_manifest,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        evaluation_suite=args.evaluation_suite,
    )
    _, dpo_summary = load_dpo_splits(
        args.dpo_manifest,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        evaluation_suite=args.evaluation_suite,
    )
    _, grpo_summary = load_grpo_splits(
        args.grpo_manifest,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        max_new_tokens=args.max_new_tokens,
        evaluation_suite=args.evaluation_suite,
    )
    manifest_paths = {
        RecordKind.SFT: args.sft_manifest,
        RecordKind.DPO: args.dpo_manifest,
        RecordKind.GRPO: args.grpo_manifest,
    }
    source_ids = {}
    for kind, manifest_path in manifest_paths.items():
        manifest = load_data_manifest(manifest_path)
        if manifest.record_kind is not kind:
            raise DataValidationError(
                f"{kind.value} manifest has record_kind={manifest.record_kind.value}"
            )
        values = set()
        for split in manifest.splits:
            validated = validate_jsonl(manifest_path.parent / split.path, kind)
            if not validated.report.ok:
                raise DataValidationError(format_validation_failure(validated.report))
            values.update(str(row["source_id"]) for row in validated.records)
        source_ids[kind] = values
    overlaps = {}
    for left, right in combinations(_POSTTRAIN_STAGES, 2):
        overlap = source_ids[left] & source_ids[right]
        name = f"{left.value}-{right.value}"
        overlaps[name] = len(overlap)
        if overlap:
            raise DataValidationError(
                f"post-training source_id overlap between {left.value} and "
                f"{right.value}: {len(overlap)} groups"
            )

    result = {
        "ok": True,
        "sequence_length": args.sequence_length,
        "max_new_tokens": args.max_new_tokens,
        "stage_source_id_overlap": overlaps,
        "sft": sft_summary,
        "dpo": dpo_summary,
        "grpo": grpo_summary,
    }
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("PASS prepared SFT, DPO, and GRPO data")
        count_fields = {
            RecordKind.SFT: "examples",
            RecordKind.DPO: "pairs",
            RecordKind.GRPO: "prompts",
        }
        for kind in _POSTTRAIN_STAGES:
            summary = result[kind.value]["splits"]
            counts = ", ".join(
                f"{split}={values[count_fields[kind]]}"
                for split, values in summary.items()
            )
            print(f"  {kind.value}: {counts}")
        print("  cross-stage source_id overlap: 0")
    return 0


def pack_tokens(args: argparse.Namespace) -> int:
    # Packing imports PyTorch, while lightweight commands such as `recipes` do not.
    from llm_lifecycle_lab.data.packing import (
        materialize_packed_pretraining_dataset,
    )

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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "recipes": list_recipes,
        "fetch": fetch_source,
        "posttrain-recipes": list_posttraining_recipes,
        "fetch-posttrain": fetch_posttraining,
        "mix": mix_sources,
        "validate": validate_source,
        "prepare": prepare_splits,
        "check-posttrain": check_posttraining_data,
        "pack": pack_tokens,
    }
    try:
        return commands[args.command](args)
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
