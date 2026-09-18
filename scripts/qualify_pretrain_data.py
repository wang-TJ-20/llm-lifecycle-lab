"""Count Base-v2 train tokens and enforce language/source composition gates."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.data.qualification import qualify_pretraining_data
from llm_lifecycle_lab.exceptions import LLMLabError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/qualify_pretrain_data.py",
        description=__doc__,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-train-tokens", type=int, default=300_000_000)
    parser.add_argument("--maximum-train-tokens", type=int, default=500_000_000)
    parser.add_argument("--minimum-language-fraction", type=float, default=0.40)
    parser.add_argument("--maximum-language-fraction", type=float, default=0.60)
    parser.add_argument("--maximum-source-fraction", type=float, default=0.50)
    parser.add_argument("--maximum-synthetic-fraction", type=float, default=0.30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        print(f"error: output exists: {args.output}", file=sys.stderr)
        return 2
    try:
        report = qualify_pretraining_data(
            args.manifest,
            args.tokenizer,
            minimum_train_tokens=args.minimum_train_tokens,
            maximum_train_tokens=args.maximum_train_tokens,
            minimum_language_fraction=args.minimum_language_fraction,
            maximum_language_fraction=args.maximum_language_fraction,
            maximum_source_fraction=args.maximum_source_fraction,
            maximum_synthetic_fraction=args.maximum_synthetic_fraction,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (LLMLabError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    status = "PASS" if report["ok"] else "FAIL"
    print(
        f"{status} train_tokens={report['train_tokens']} "
        f"en={report['languages']['en']['fraction']:.2%} "
        f"zh={report['languages']['zh']['fraction']:.2%} "
        f"synthetic={report['synthetic_fraction']:.2%}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
