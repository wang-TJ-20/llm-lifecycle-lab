"""准备可复现的预训练数据。

本脚本提供下载、混合、校验、切分与打包子命令。
切分后需单独运行 ``train_tokenizer.py`` 训练分词器，再执行 ``pack``。
各步骤直接调用 ``src/llm_lifecycle_lab/data`` 下对应模块的算法实现。

Prepare reproducible pretraining data.

This script provides fetch, mix, validate, prepare, and pack subcommands.
After prepare, run ``train_tokenizer.py`` separately before pack.
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
from llm_lifecycle_lab.data.prepare import prepare_dataset
from llm_lifecycle_lab.data.public import (
    available_public_recipes,
    load_public_source_manifest,
    materialize_public_dataset,
)
from llm_lifecycle_lab.data.schemas import format_validation_failure, validate_jsonl
from llm_lifecycle_lab.data.split import SplitRatios
from llm_lifecycle_lab.exceptions import LLMLabError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/data.py",
        description="Fetch, validate, prepare, and pack pretraining data.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    recipes = commands.add_parser("recipes", help="list built-in data recipes")
    recipes.add_argument("--json", action="store_true", dest="as_json")

    fetch = commands.add_parser("fetch", help="download one pinned source recipe")
    fetch.add_argument("--recipe", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--accept-license", required=True)

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
        "mix": mix_sources,
        "validate": validate_source,
        "prepare": prepare_splits,
        "pack": pack_tokens,
    }
    try:
        return commands[args.command](args)
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
