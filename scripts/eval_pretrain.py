"""在 dev 或 test 数据划分上评测 Native 预训练模型的检查点。

Evaluate a Native Pretrain checkpoint on the dev or test split.
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

from llm_lifecycle_lab.config import load_run_config
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.training.pretrain import evaluate_native_pretraining


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/eval_pretrain.py",
        description="Evaluate a Native checkpoint on held-out data.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_run_config(args.config)
        report = evaluate_native_pretraining(
            config,
            checkpoint=args.checkpoint,
            split=args.split,
            workdir=Path.cwd(),
        )
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

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
            f"bits_per_byte={report[f'eval_{language}_bits_per_byte']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
