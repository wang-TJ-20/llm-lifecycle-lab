"""从 Native SFT/DPO checkpoint 开始可验证奖励 GRPO，或恢复同一次训练。

Run or resume Native GRPO with deterministic programmatic reward verifiers.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.config import load_run_config
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.training.grpo import run_native_grpo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/train_grpo.py", description=__doc__
    )
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-id")
    group.add_argument("--resume-run")
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run = run_native_grpo(
            load_run_config(args.config),
            run_id=args.run_id,
            resume_run=args.resume_run,
            resume_checkpoint=args.resume_checkpoint,
        )
        print(json.dumps(asdict(run.result), ensure_ascii=False, indent=2))
    except (LLMLabError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
