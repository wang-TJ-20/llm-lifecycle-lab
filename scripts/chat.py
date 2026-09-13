"""本地模型续写和多轮对话；Base 默认使用续写模式。

Continue text or chat with local Native/HF models; Base defaults to completion.
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

from llm_lifecycle_lab.chat import ChatSession
from llm_lifecycle_lab.evaluation.native import NativeEvaluator
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.model.protocol import GenerationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python scripts/chat.py", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("native", "hf"), default="native")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument(
        "--mode", choices=("auto", "completion", "chat"), default="auto"
    )
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--system")
    parser.add_argument(
        "--history-policy", choices=("error", "drop-oldest"), default="error"
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.threads <= 0:
            raise ValueError("--threads must be positive")
        if args.as_json and not args.prompt:
            raise ValueError("--json requires --prompt")
        torch.set_num_threads(args.threads)
        if args.backend == "hf":
            if args.tokenizer:
                raise ValueError("HF uses its bound tokenizer; omit --tokenizer")
            from llm_lifecycle_lab.interop.evaluation import HFEvaluator

            evaluator = HFEvaluator(
                args.checkpoint, base_model=args.base_model, device=args.device
            )
        else:
            if args.base_model:
                raise ValueError("--base-model is only for HF adapters")
            evaluator = NativeEvaluator(
                args.checkpoint, tokenizer_dir=args.tokenizer, device=args.device
            )
        session = ChatSession(
            evaluator,
            mode=args.mode,
            system=args.system,
            history_policy=args.history_policy,
            generation=GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                do_sample=args.sample,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            ),
        )
        if args.prompt:
            results = [session.respond(prompt) for prompt in args.prompt]
            if args.as_json:
                print(json.dumps(results, ensure_ascii=False, indent=2))
            else:
                for result in results:
                    print(result["text"])
        else:
            print(f"mode={session.mode}; /reset 清空对话，/exit 退出。")
            if evaluator.identity["stage"] == "pretrain":
                print("当前是 Base 模型；对话模板不代表已具备指令能力。")
            while True:
                try:
                    prompt = input("you> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if prompt.strip() == "/exit":
                    break
                if prompt.strip() == "/reset":
                    session.reset()
                    continue
                try:
                    result = session.respond(prompt)
                    print(f"model> {result['text']}")
                    if result["dropped_turns"]:
                        print(f"已丢弃 {result['dropped_turns']} 个旧回合。")
                except LLMLabError as exc:
                    print(f"error: {exc}", file=sys.stderr)
    except (LLMLabError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
