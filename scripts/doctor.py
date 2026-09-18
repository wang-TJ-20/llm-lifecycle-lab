"""检查运行环境；指定配置后，还会验证一个真实训练批次。

Check the environment and, when configured, one real training batch.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.config import load_run_config, with_init_checkpoint
from llm_lifecycle_lab.doctor.preflight import available_profiles, run_doctor
from llm_lifecycle_lab.doctor.result import CheckStatus, DoctorReport
from llm_lifecycle_lab.exceptions import LLMLabError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/doctor.py",
        description="Check the environment and data before training.",
    )
    parser.add_argument("--profile", choices=available_profiles())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def print_report(report: DoctorReport) -> None:
    for check in report.checks:
        label = {
            CheckStatus.PASS: "PASS",
            CheckStatus.WARN: "WARN",
            CheckStatus.FAIL: "FAIL",
        }[check.status]
        print(f"{label:4} {check.name}: {check.message}")

    counts = report.counts
    print(
        f"Summary: {counts['pass']} passed, "
        f"{counts['warn']} warnings, {counts['fail']} failed"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_run_config(args.config) if args.config else None
        if args.init_checkpoint is not None:
            if config is None:
                raise ValueError("--init-checkpoint requires --config")
            config = with_init_checkpoint(config, args.init_checkpoint)
        report = run_doctor(
            profile=args.profile,
            config=config,
            workdir=Path.cwd(),
        )
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(report.to_json())
    else:
        print_report(report)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
