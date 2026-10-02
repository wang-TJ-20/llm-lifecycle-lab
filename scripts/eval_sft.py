"""在冻结的 dev 或 test 划分上评测 Native SFT checkpoint。

Evaluate a Native SFT checkpoint on the frozen dev or test split.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.config import load_run_config
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.training.sft import evaluate_native_sft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/eval_sft.py",
        description="Evaluate assistant-only loss on frozen SFT data.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_native_sft(
            load_run_config(args.config),
            checkpoint=args.checkpoint,
            split=args.split,
            workdir=Path.cwd(),
        )
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(
        f"checkpoint_step={report['checkpoint_step']} split={report['split']} "
        f"examples={report['examples']} loss={report['eval_loss']:.6f} "
        f"perplexity={report['eval_perplexity']:.4f}"
    )
    for language in ("en", "zh"):
        key = f"eval_{language}_loss"
        if key in report:
            print(
                f"{language}: loss={report[key]:.6f} "
                f"perplexity={report[f'eval_{language}_perplexity']:.4f} "
                f"tokens={int(report[f'eval_{language}_tokens'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
