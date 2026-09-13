"""启动本地、非流式的最小 OpenAI-compatible 推理 API。

Serve a local strict subset of the OpenAI completions and chat API.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.evaluation.native import NativeEvaluator
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.serving import OpenAIService, create_openai_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/serve_openai.py",
        description=__doc__,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("native", "hf"), default="native")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--default-max-tokens", type=int, default=48)
    parser.add_argument("--max-request-bytes", type=int, default=1024 * 1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = None
    try:
        if args.threads <= 0:
            raise ValueError("--threads must be positive")
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be between 1 and 65535")
        if args.backend == "hf":
            if args.tokenizer:
                raise ValueError("HF uses its bound tokenizer; omit --tokenizer")
            from llm_lifecycle_lab.interop.evaluation import HFEvaluator

            evaluator = HFEvaluator(
                args.checkpoint,
                base_model=args.base_model,
                device=args.device,
            )
        else:
            if args.base_model:
                raise ValueError("--base-model is only for HF adapters")
            evaluator = NativeEvaluator(
                args.checkpoint,
                tokenizer_dir=args.tokenizer,
                device=args.device,
            )
        torch.set_num_threads(args.threads)
        model_id = args.model_id or args.checkpoint.resolve().name
        service = OpenAIService(
            evaluator,
            model_id=model_id,
            default_max_tokens=args.default_max_tokens,
            max_request_bytes=args.max_request_bytes,
        )
        server = create_openai_server(args.host, args.port, service)
        host, port = server.server_address[:2]
        print(
            f"serving model={model_id} stage={evaluator.identity['stage']} "
            f"at http://{host}:{port}/v1",
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (LLMLabError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if server is not None:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
