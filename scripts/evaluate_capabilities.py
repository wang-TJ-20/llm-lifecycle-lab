"""运行统一双语能力评测，或比较同协议的阶段成绩单。

Evaluate bilingual capabilities or compare reports with identical protocols.
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

from llm_lifecycle_lab.evaluation.report import compare_files
from llm_lifecycle_lab.evaluation.runner import evaluate_capabilities
from llm_lifecycle_lab.exceptions import LLMLabError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/evaluate_capabilities.py", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="评测本地 checkpoint 或发布包")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--tokenizer", type=Path)
    run.add_argument("--backend", choices=("native", "hf"), default="native")
    run.add_argument("--base-model", type=Path, help="base snapshot for HF adapters")
    run.add_argument(
        "--suite", type=Path, default=Path("configs/evaluation/lifecycle-v1.yaml")
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--max-new-tokens", type=int, default=32)
    run.add_argument(
        "--prompt-protocol", choices=("plain-v1", "native-chat-v1"), default="plain-v1"
    )
    run.add_argument("--pretrain-manifest", type=Path)
    run.add_argument("--baseline", type=Path)
    compare = sub.add_parser("compare", help="按参数顺序生成纵向成绩单")
    compare.add_argument("--reports", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "compare":
            result = compare_files(args.reports, args.output)
        else:
            if args.threads <= 0:
                raise ValueError("--threads must be positive")
            torch.set_num_threads(args.threads)
            result = evaluate_capabilities(
                checkpoint=args.checkpoint,
                tokenizer_dir=args.tokenizer,
                suite_path=args.suite,
                output=args.output,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                prompt_protocol=args.prompt_protocol,
                pretrain_manifest=args.pretrain_manifest,
                baseline_path=args.baseline,
                backend=args.backend,
                base_model=args.base_model,
            )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "protocol_sha256": result["protocol_sha256"],
                    "metrics": len(result["metrics"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
