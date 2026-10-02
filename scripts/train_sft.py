"""从本地 Base 权重开始 SFT，或恢复同一次 SFT 的完整训练状态。

Initialize SFT from local Base weights or resume the same SFT run.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.config import load_run_config
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.training.sft import run_native_sft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/train_sft.py", description=__doc__
    )
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-id")
    group.add_argument("--resume-run")
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def print_metric(metric: Mapping[str, object]) -> None:
    step = metric.get("step", "?")
    if "train_loss" in metric:
        message = f"step={step} train_loss={float(metric['train_loss']):.6f}"
        if "eval_loss" in metric:
            message += f" eval_loss={float(metric['eval_loss']):.6f}"
        print(message)
    elif "eval_loss" in metric:
        print(
            f"{metric.get('event', 'evaluation')} step={step} "
            f"eval_loss={float(metric['eval_loss']):.6f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_run_config(args.config)
        run = run_native_sft(
            config,
            run_id=args.run_id,
            resume_run=args.resume_run,
            resume_checkpoint=args.resume_checkpoint,
            workdir=Path.cwd(),
            metric_callback=print_metric,
        )
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Completed SFT run {run.artifacts.run_id}")
    print(f"  parameters: {run.parameter_count:,}")
    print(f"  steps: {run.result.global_step}")
    print(f"  supervised_tokens: {run.result.tokens_seen}")
    print(f"  target_train_tokens: {run.result.target_train_tokens}")
    print(f"  target_token_coverage: {run.result.target_token_coverage:.2%}")
    print(f"  final_loss: {run.result.final_loss:.6f}")
    print(f"  checkpoint: {run.result.final_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
