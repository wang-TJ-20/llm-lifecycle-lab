"""从本地 Base 权重开始 SFT，或恢复同一次 SFT 的完整训练状态。

Initialize SFT from local Base weights or resume the same SFT run.
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
from llm_lifecycle_lab.training.sft import run_native_sft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/train_sft.py", description=__doc__
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
        run = run_native_sft(
            load_run_config(args.config),
            run_id=args.run_id,
            resume_run=args.resume_run,
            resume_checkpoint=args.resume_checkpoint,
        )
        print(json.dumps(asdict(run.result), ensure_ascii=False, indent=2))
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
