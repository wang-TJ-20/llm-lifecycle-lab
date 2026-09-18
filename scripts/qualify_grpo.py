"""Run the fixed train-only GRPO rollout qualification without optimization."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.config import load_run_config, with_init_checkpoint
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.training.qualification import qualify_native_grpo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/qualify_grpo.py",
        description=__doc__,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--source-groups", type=int, default=64)
    parser.add_argument("--minimum-mixed-group-fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.output / "qualification.json"
    if args.output.exists():
        print(f"error: output exists: {args.output}", file=sys.stderr)
        return 2
    try:
        config = with_init_checkpoint(
            load_run_config(args.config),
            args.init_checkpoint,
        )
        report = qualify_native_grpo(
            config,
            source_groups=args.source_groups,
            minimum_mixed_group_fraction=(
                args.minimum_mixed_group_fraction
            ),
            workdir=Path.cwd(),
        )
        args.output.mkdir(parents=True)
        report_path.write_text(
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
        f"{status} mixed={report['mixed_reward_groups']}/"
        f"{report['prompt_groups']} "
        f"success_rate={report['success_rate']:.2%} "
        f"parse_failures={report['verifier_parse_failures']}"
    )
    print(f"  report: {report_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
