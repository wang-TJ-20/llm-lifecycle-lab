"""用随机 token 实践模型构建、前向、反向、参数更新、KV Cache 和保存加载。

Practice model construction, forward/backward, one update, caching, and save/load.
Random tokens demonstrate mechanics, not language quality.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

import torch
import torch.nn.functional as F

from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.model.native import NativeTransformer, load_native_model_config
from llm_lifecycle_lab.model.protocol import GenerationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/model_experiment.py",
        description="Run one CPU learning step and check model mechanics.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/models/smoke-10m.yaml")
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_native_model_config(args.config)
        if args.batch_size <= 0 or args.seed < 0:
            raise ValueError("batch-size must be positive and seed non-negative")
        if not 2 <= args.sequence_length < config.max_sequence_length:
            raise ValueError("sequence-length must be in [2, max_sequence_length)")
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("learning-rate must be finite and positive")

        # 1. 随机初始化模型和输入。Initialize the model and integer token IDs.
        torch.manual_seed(args.seed)
        model = NativeTransformer(config)
        input_ids = torch.randint(
            0, config.vocab_size, (args.batch_size, args.sequence_length)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        # 2. 前向：[B, T] -> [B, T, V]。Predict logits for every input position.
        output = model(input_ids=input_ids)

        # 3. 位置 t 预测 t+1；只在这里错位一次。Align next-token targets once.
        predictions = output.logits[:, :-1, :].reshape(-1, config.vocab_size)
        targets = input_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(predictions, targets)

        # 4. 反向计算梯度，再更新参数。Backward first, then update parameters.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        observed_token = int(input_ids[0, 0])
        before = model.token_embedding.weight[observed_token].detach().clone()
        optimizer.step()
        with torch.no_grad():
            update = (model.token_embedding.weight[observed_token] - before).abs().max()
        if not torch.isfinite(loss) or not torch.isfinite(update) or update == 0:
            raise RuntimeError(
                "the learning step did not produce a finite weight update"
            )

        # 5. KV Cache 续写应等价于完整前向。Cached suffix must match full forward.
        model.eval()
        split = args.sequence_length // 2
        with torch.inference_mode():
            full = model(input_ids=input_ids).logits
            prefix = model(input_ids=input_ids[:, :split], use_cache=True)
            suffix = model(
                input_ids=input_ids[:, split:], cache=prefix.cache, use_cache=True
            )
            torch.testing.assert_close(
                full[:, split:], suffix.logits, atol=1e-5, rtol=1e-5
            )

            # 6. 临时保存并恢复，退出自动清理。Check save/load, then clean up.
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "model"
                model.save(checkpoint)
                restored = NativeTransformer(config).eval()
                restored.load(checkpoint)
                torch.testing.assert_close(
                    full, restored(input_ids=input_ids).logits, atol=0, rtol=0
                )

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=None,
            config=GenerationConfig(
                max_new_tokens=min(4, config.max_sequence_length - args.sequence_length)
            ),
        )
        result = {
            "model_id": config.model_id,
            "parameters": model.parameter_count,
            "input_shape": list(input_ids.shape),
            "logits_shape": list(output.logits.shape),
            "supervised_tokens": targets.numel(),
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "weight_update_max": float(update),
            "cache_key_shape": list(suffix.cache[0][0].shape),
            "cache_matches_full": True,
            "checkpoint_matches_full": True,
            "generated_shape": list(generated.token_ids.shape),
        }
    except (LLMLabError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        for name, value in result.items():
            print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
