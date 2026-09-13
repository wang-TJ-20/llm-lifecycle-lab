"""导出标准 HF 模型并验证 CPU 一致性。

Export a Native checkpoint to standard HF files with mandatory parity checks.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.interop.export import export_native_hf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/export_hf.py", description=__doc__
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.threads <= 0:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(args.threads)
        result = export_native_hf(
            args.checkpoint, args.output, tokenizer_dir=args.tokenizer
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "architecture": result["architecture"],
                    "verified": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except (LLMLabError, OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
