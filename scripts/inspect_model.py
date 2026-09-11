"""查看 Native 模型的结构维度，以及不同词表大小对应的参数量。

Inspect Native model dimensions and parameter budgets for different vocabulary sizes.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.model.native.config import load_native_model_config
from llm_lifecycle_lab.model.native.transformer import NativeTransformer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/inspect_model.py",
        description="Inspect a Native model configuration and parameter budget.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--compare-vocab-size",
        type=int,
        action="append",
        default=[],
        help="compare the parameter budget with another vocabulary size",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_native_model_config(args.config)
        model = NativeTransformer(config)
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    comparison_sizes = tuple(dict.fromkeys(args.compare_vocab_size))
    vocab_budget = [
        {
            "vocab_size": vocab_size,
            "parameter_count": config.parameter_count_for_vocab_size(vocab_size),
            "token_parameter_count": config.token_parameter_count_for_vocab_size(
                vocab_size
            ),
            "token_parameter_share": config.token_parameter_share_for_vocab_size(
                vocab_size
            ),
        }
        for vocab_size in comparison_sizes
    ]
    result = {
        "model_id": config.model_id,
        "parameter_count": model.parameter_count,
        "vocab_size": config.vocab_size,
        "token_parameter_count": config.token_parameter_count_for_vocab_size(
            config.vocab_size
        ),
        "token_parameter_share": config.token_parameter_share_for_vocab_size(
            config.vocab_size
        ),
        "layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "attention_heads": config.num_attention_heads,
        "key_value_heads": config.num_key_value_heads,
        "qk_norm": config.qk_norm,
        "max_sequence_length": config.max_sequence_length,
        "vocab_budget": vocab_budget,
    }

    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print(f"model_id: {config.model_id}")
    print(f"parameters: {model.parameter_count:,}")
    print(
        f"layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"heads={config.num_attention_heads} "
        f"kv_heads={config.num_key_value_heads} qk_norm={config.qk_norm}"
    )
    print(
        f"token_parameters={result['token_parameter_count']:,} "
        f"share={result['token_parameter_share']:.2%}"
    )
    for item in vocab_budget:
        print(
            f"vocab={item['vocab_size']}: "
            f"parameters={item['parameter_count']:,} "
            f"token_parameters={item['token_parameter_count']:,} "
            f"share={item['token_parameter_share']:.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
