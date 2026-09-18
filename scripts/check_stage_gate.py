"""Evaluate pre-registered Base, SFT, DPO, or GRPO stage gates."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.gates import (
    check_base_gate,
    check_dpo_gate,
    check_grpo_gate,
    check_sft_gate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/check_stage_gate.py",
        description=__doc__,
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    base = stages.add_parser("base")
    _common_run_output(base)
    base.add_argument("--dev-evaluation", type=Path, required=True)
    base.add_argument("--test-evaluation", type=Path, required=True)
    base.add_argument("--baseline-test-evaluation", type=Path, required=True)

    sft = stages.add_parser("sft")
    _common_run_output(sft)
    _capability_arguments(sft)
    sft.add_argument("--data-check", type=Path, required=True)

    dpo = stages.add_parser("dpo")
    _common_run_output(dpo)
    _capability_arguments(dpo)

    grpo = stages.add_parser("grpo")
    _common_run_output(grpo)
    _capability_arguments(grpo)
    return parser


def _common_run_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _capability_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capability-report", type=Path, required=True)
    parser.add_argument("--parent-capability-report", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        print(f"error: output exists: {args.output}", file=sys.stderr)
        return 2
    try:
        if args.stage == "base":
            report = check_base_gate(
                args.run_dir,
                dev_evaluation=args.dev_evaluation,
                test_evaluation=args.test_evaluation,
                baseline_test_evaluation=args.baseline_test_evaluation,
            )
        elif args.stage == "sft":
            report = check_sft_gate(
                args.run_dir,
                capability_report=args.capability_report,
                parent_capability_report=args.parent_capability_report,
                data_check=args.data_check,
            )
        elif args.stage == "dpo":
            report = check_dpo_gate(
                args.run_dir,
                capability_report=args.capability_report,
                parent_capability_report=args.parent_capability_report,
            )
        else:
            report = check_grpo_gate(
                args.run_dir,
                capability_report=args.capability_report,
                parent_capability_report=args.parent_capability_report,
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
    print(f"{'PASS' if report['ok'] else 'FAIL'} {args.stage} stage gate")
    for check in report["checks"]:
        print(
            f"  {'PASS' if check['ok'] else 'FAIL'} {check['name']}: "
            f"{check['actual']}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
