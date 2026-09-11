"""按版本化参考规范检查基线输入、训练前条件或已完成的实验。

Verify baseline inputs, training prerequisites, or a completed reference run.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.doctor.result import CheckStatus
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.reference import verify_reference_inputs, verify_reference_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/verify_reference.py",
        description="Check frozen inputs, runtime prerequisites, or a completed run.",
    )
    parser.add_argument("--spec", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path, help="verify a completed run")
    mode.add_argument(
        "--preflight", action="store_true", help="check inputs and runtime"
    )
    mode.add_argument(
        "--inputs-only",
        action="store_true",
        help="check inputs only; does not certify CUDA readiness or model quality",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.run is not None:
            report = verify_reference_run(args.spec, args.run, workdir=Path.cwd())
        else:
            report = verify_reference_inputs(
                args.spec, workdir=Path.cwd(), check_runtime=args.preflight
            )
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(report.to_json())
        return report.exit_code

    print(f"Scope: {report.scope}")
    for check in report.checks:
        label = "PASS" if check.status is CheckStatus.PASS else "FAIL"
        print(f"{label:4} {check.name}: {check.message}")
    counts = report.counts
    print(
        f"Summary: {counts['pass']} passed, "
        f"{counts['warn']} warnings, {counts['fail']} failed"
    )
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
