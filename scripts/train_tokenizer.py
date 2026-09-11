"""训练 Native 字节级 BPE 分词器。

本脚本展示分词器训练的入口流程：
1. 读取预处理数据清单。
2. 仅使用 train 数据划分训练字节级 BPE 分词器。
3. 保存 ``tokenizer.json`` 和用于复现的清单。

Train the Native byte-level BPE tokenizer.

This script is the entry point for the tokenizer training workflow:
1. Read the prepared-data manifest.
2. Train a byte-level BPE tokenizer from its train split only.
3. Save ``tokenizer.json`` and a reproducibility manifest.

核心实现 / Implementation:
``src/llm_lifecycle_lab/tokenizer/native.py::train_native_tokenizer``.
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
from llm_lifecycle_lab.tokenizer.native import train_native_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/train_tokenizer.py",
        description="Train a byte-level BPE tokenizer from prepared data.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--min-frequency", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = train_native_tokenizer(
            args.manifest,
            args.output,
            tokenizer_id=args.tokenizer_id,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Trained tokenizer {manifest.tokenizer_id}")
    print(f"  path: {args.output}")
    print(f"  vocab_size: {manifest.vocab_size}")
    print(f"  sha256: {manifest.content_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
