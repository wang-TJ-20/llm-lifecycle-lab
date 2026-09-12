"""用固定提示集生成可复核的预训练续写样本。

Generate reproducible continuations from a Native pretraining checkpoint.
Decoding uses deterministic greedy search and records all generation settings.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.config import config_sha256, load_run_config
from llm_lifecycle_lab.contracts import CheckpointMetadata, ModelRoute, Stage
from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError, LLMLabError
from llm_lifecycle_lab.model.native import (
    NativeTransformer,
    load_native_model_config,
)
from llm_lifecycle_lab.model.protocol import GenerationConfig
from llm_lifecycle_lab.tokenizer import NativeTokenizer

FIXED_PROMPTS: tuple[str, ...] = (
    "Once upon a time",
    "The little girl",
    "One day, a boy",
    "There was a dragon",
    "从前",
    "在中国",
    "这个故事",
    "北京是",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/generate_pretrain.py",
        description="Generate fixed-prompt samples from a Native checkpoint.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="override the fixed prompt set; repeat for multiple prompts",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _load_checkpoint_metadata(path: Path) -> CheckpointMetadata:
    try:
        value = json.loads(
            (path / "checkpoint_metadata.json").read_text(encoding="utf-8")
        )
        return CheckpointMetadata.from_dict(value)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid checkpoint metadata under {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_run_config(args.config)
        if (
            config.model_route is not ModelRoute.NATIVE
            or config.stage is not Stage.PRETRAIN
        ):
            raise ConfigError("generation requires a native pretrain config")

        root = Path.cwd()
        model_config_path = Path(str(config.model["config"]))
        tokenizer_path = Path(str(config.model["tokenizer"]))
        checkpoint_path = Path(args.checkpoint)
        if not model_config_path.is_absolute():
            model_config_path = root / model_config_path
        if not tokenizer_path.is_absolute():
            tokenizer_path = root / tokenizer_path
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path

        model_config = load_native_model_config(model_config_path)
        tokenizer = NativeTokenizer.from_directory(tokenizer_path)
        metadata = _load_checkpoint_metadata(checkpoint_path)
        expected = {
            "model_route": (metadata.model_route, config.model_route),
            "stage": (metadata.stage, config.stage),
            "tokenizer_sha256": (
                metadata.tokenizer_sha256,
                tokenizer.manifest.content_sha256,
            ),
            "config_sha256": (metadata.config_sha256, config_sha256(config)),
        }
        mismatches = [
            name for name, (actual, wanted) in expected.items() if actual != wanted
        ]
        if mismatches:
            raise ArtifactError(
                "generation checkpoint is incompatible: " + ", ".join(mismatches)
            )
        if tokenizer.vocab_size != model_config.vocab_size:
            raise ConfigError("tokenizer and model vocabulary sizes do not match")

        model = NativeTransformer(model_config)
        model.load(checkpoint_path / "model")
        model.set_training(False)

        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            seed=args.seed,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompts = tuple(args.prompts) if args.prompts else FIXED_PROMPTS
        torch.manual_seed(args.seed)

        results = []
        for prompt in prompts:
            input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)])
            output = model.generate(
                input_ids=input_ids,
                attention_mask=None,
                config=generation_config,
            )
            generated = output.token_ids[0, input_ids.shape[1] :].tolist()
            results.append(
                {
                    "prompt": prompt,
                    "prompt_tokens": int(input_ids.shape[1]),
                    "generated_tokens": int(output.generated_tokens),
                    "stop_reason": output.stop_reason,
                    "generated_token_ids": generated,
                    "continuation": tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                    ),
                }
            )
    except (LLMLabError, ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = {
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "seed": args.seed,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "results": results,
    }

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    for item in results:
        print(f"prompt: {item['prompt']}")
        print(f"  continuation: {item['continuation']}")
        print(f"  stop_reason: {item['stop_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
