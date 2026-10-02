"""验收正式 SFT run 的开发集结果与一次性 sealed test。

Verify formal SFT development results and the one-time sealed test.
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
from llm_lifecycle_lab.sft_run_gate import verify_sft_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/verify_sft_run.py",
        description="Check a completed Native-60M SFT run.",
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dev",
        action="store_true",
        help="require training metrics and the complete frozen dev report",
    )
    mode.add_argument(
        "--sealed-test",
        action="store_true",
        help="also require the one-time complete frozen test report",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_sft_run(
            args.spec,
            args.run,
            workdir=Path.cwd(),
            sealed_test=args.sealed_test,
        )
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(report.to_json())
        return report.exit_code

    print(f"Gate: {report.gate_id}")
    print(f"Scope: {report.scope}")
    print(f"Run: {report.run_path}")
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
