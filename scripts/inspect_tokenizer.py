"""使用训练好的 Native 分词器，对一段文本进行编码和解码。

Encode and decode one text sample with a trained Native tokenizer.
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
from llm_lifecycle_lab.tokenizer.native import NativeTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/inspect_tokenizer.py",
        description="Inspect how a Native tokenizer encodes and decodes text.",
    )
    parser.add_argument("path", type=Path, help="directory containing tokenizer.json")
    parser.add_argument("--text", required=True, help="text sample to tokenize")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tokenizer = NativeTokenizer.from_directory(args.path)
        token_ids = tokenizer.encode(args.text, add_bos=True, add_eos=True)
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"token_ids: {token_ids}")
    print(f"tokens: {len(token_ids)}")
    print(f"decoded: {tokenizer.decode(token_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
