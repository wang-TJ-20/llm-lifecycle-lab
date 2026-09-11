"""训练 Native 因果语言模型，或从检查点恢复训练。

本脚本读取训练配置，调用预训练流程，并输出进度和结果。
训练流程编排位于 ``src/llm_lifecycle_lab/training/pretrain.py``；
训练循环位于 ``src/llm_lifecycle_lab/training/engine.py``。

Train the Native causal language model or resume from a checkpoint.

This script loads a pipeline config, calls pretraining, and reports progress
and results.
Run orchestration lives in ``src/llm_lifecycle_lab/training/pretrain.py``;
the training loop lives in ``src/llm_lifecycle_lab/training/engine.py``.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.config import load_run_config
from llm_lifecycle_lab.exceptions import ConfigError, LLMLabError
from llm_lifecycle_lab.reference import verify_reference_inputs
from llm_lifecycle_lab.training.pretrain import run_native_pretraining


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/train_pretrain.py",
        description="Train or resume a Native pretraining run.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume-run")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--reference-spec",
        type=Path,
        help="check frozen inputs and runtime before creating or resuming a run",
    )
    return parser


def print_metric(metric: Mapping[str, object]) -> None:
    """Print the small set of metrics useful while following a training run."""

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
    elif "eval_loss" in metric:
        print(
            f"{metric.get('event', 'evaluation')} step={step} "
            f"eval_loss={float(metric['eval_loss']):.6f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_run_config(args.config)
        if args.reference_spec is not None:
            report = verify_reference_inputs(
                args.reference_spec, config=config, workdir=Path.cwd()
            )
            if report.has_failures:
                failures = "; ".join(
                    f"{check.name}: {check.message}"
                    for check in report.checks
                    if check.status.value == "fail"
                )
                raise ConfigError(f"reference preflight failed: {failures}")
        run = run_native_pretraining(
            config,
            run_id=args.run_id,
            resume_run=args.resume_run,
            resume_checkpoint=args.resume_checkpoint,
            workdir=Path.cwd(),
            metric_callback=print_metric,
        )
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

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


if __name__ == "__main__":
    raise SystemExit(main())
