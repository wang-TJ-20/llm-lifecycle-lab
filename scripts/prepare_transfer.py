"""下载固定 revision 的 Qwen 快照，生成本地 hash 清单与 SFT 示例配置。

Prepare an explicitly pinned Qwen snapshot and a local training configuration.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.interop.transfer import prepare_qwen_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/prepare_transfer.py", description=__doc__
    )
    parser.add_argument(
        "--revision", required=True, help="40-character upstream commit SHA"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare_qwen_snapshot(
            args.output, revision=args.revision, offline=args.offline
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "revision": result["revision"],
                    "config": str(args.output / "sft.example.yaml"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except (LLMLabError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
