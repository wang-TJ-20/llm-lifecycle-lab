"""校验冻结的 SFT 公开数据契约与物化产物。

Verify the frozen public-data contract and materialized artifacts for SFT.
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
from llm_lifecycle_lab.sft_data_gate import verify_sft_data_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/verify_sft_data.py",
        description="Check the frozen Native-60M SFT public-data boundary.",
    )
    parser.add_argument("--spec", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--contract-only",
        action="store_true",
        help="verify the frozen recipe and gate without local data",
    )
    mode.add_argument(
        "--inputs-only",
        action="store_true",
        help="verify Stage 0 plus materialized source, splits and tokenizer",
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="verify all inputs and require a clean current Git worktree",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_sft_data_gate(
            args.spec,
            workdir=Path.cwd(),
            check_inputs=args.inputs_only or args.preflight,
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
