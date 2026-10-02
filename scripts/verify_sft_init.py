"""校验 SFT 是否从冻结的 canonical Base 权重开始。

Verify that SFT starts from the frozen canonical Base weights.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.sft_initialization import verify_sft_initialization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/verify_sft_init.py",
        description="Check the frozen Base-v1 initialization boundary for SFT.",
    )
    parser.add_argument("--spec", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--inputs-only",
        action="store_true",
        help="verify frozen Base artifacts and evidence",
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="verify frozen inputs and require a clean current Git worktree",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_sft_initialization(
            args.spec,
            workdir=Path.cwd(),
            check_git=args.preflight,
        )
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(report.to_json())
        return report.exit_code

    print(f"Gate: {report.gate_id}")
    print(f"Scope: {report.scope}")
    for check in report.checks:
        label = check.status.value.upper()
        print(f"{label:4} {check.name}: {check.message}")
    counts = report.counts
    print(
        f"Summary: {counts['pass']} passed, "
        f"{counts['warn']} warnings, {counts['fail']} failed"
    )
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
