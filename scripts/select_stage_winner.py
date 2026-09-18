"""Select a stage winner only from candidates whose hard gate passed."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.gates import select_stage_winner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/select_stage_winner.py",
        description=__doc__,
    )
    parser.add_argument("--stage", choices=("sft", "dpo", "grpo"), required=True)
    parser.add_argument(
        "--candidate",
        nargs=3,
        action="append",
        required=True,
        metavar=("RUN_DIR", "CAPABILITY_REPORT", "GATE_REPORT"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        print(f"error: output exists: {args.output}", file=sys.stderr)
        return 2
    try:
        selection = select_stage_winner(args.stage, args.candidate)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                selection,
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
    winner = selection["winner"]
    print(f"PASS {args.stage} winner: {winner['run_dir']}")
    print(f"  checkpoint: {winner['final_checkpoint']}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
