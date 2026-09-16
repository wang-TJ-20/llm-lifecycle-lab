"""Sample the SFT policy for on-policy DPO pairs and solvable GRPO prompts.

Two failure modes from the first GPU round motivate this script:

* Template-authored DPO pairs were already separable by the frozen SFT
  reference, so DPO had almost nothing left to teach and kept inflating the
  margin on pairs it had already won. Sampling the policy and keeping its real
  mistakes as ``rejected`` puts the pairs on the model's current ability
  boundary, and uses the model's own phrasing so format/style is no longer a
  confound between chosen and rejected.
* GRPO groups were dominated by ``all-0`` rewards: the policy could not solve
  the prompt at all, so the group advantage was exactly zero and no gradient
  signal survived. Measuring each prompt's empirical success rate lets GRPO
  train only on prompts with ``0 < success_rate < 1``.

Both outputs come from a single sampling pass, so the DPO pairs and the GRPO
filter describe the same policy state.

Generate on-policy DPO pairs from real policy mistakes, and keep only the GRPO
prompts whose empirical success rate is strictly between 0 and 1.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.data.grpo import reward_response, rollout_seed
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.model.native import NativeTransformer, load_native_model_config
from llm_lifecycle_lab.model.protocol import GenerationConfig
from llm_lifecycle_lab.tokenizer import NativeTokenizer
from llm_lifecycle_lab.tokenizer.native import SPECIAL_TOKENS

DATA_LICENSE = "Apache-2.0"


@dataclass(frozen=True)
class _Sampled:
    prompt_record: dict[str, Any]
    outputs: list[str]
    rewards: list[float]


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _encode_prompt(tokenizer: NativeTokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    return list(tokenizer.encode_chat(messages))


def _sample_policy(
    model: NativeTransformer,
    tokenizer: NativeTokenizer,
    records: list[dict[str, Any]],
    *,
    samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    batch_size: int,
) -> list[_Sampled]:
    results: list[_Sampled] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        for record in chunk:
            prompt_ids = _encode_prompt(tokenizer, record["prompt"])
            outputs: list[str] = []
            rewards: list[float] = []
            for index in range(samples):
                output = model.generate(
                    input_ids=torch.tensor([prompt_ids], device=device),
                    attention_mask=None,
                    config=GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        seed=rollout_seed(record["id"], index),
                        eos_token_id=tokenizer.chat_end_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    ),
                )
                sequence = output.token_ids[0].tolist()
                response = sequence[len(prompt_ids) :]
                content = (
                    response[:-1]
                    if response and response[-1] == tokenizer.chat_end_token_id
                    else response
                )
                text = tokenizer.decode(content, skip_special_tokens=False)
                outputs.append(text)
                rewards.append(
                    reward_response(
                        text,
                        record["answer"],
                        record["metadata"]["verifier"],
                    )
                )
            results.append(
                _Sampled(prompt_record=record, outputs=outputs, rewards=rewards)
            )
        if start + batch_size >= len(records) or start % (batch_size * 20) == 0:
            print(f"sampled {min(start + batch_size, len(records))}/{len(records)}")
    return results


def _contains_control_token(text: str) -> bool:
    """Reject degenerate samples that leaked special tokens into the body.

    A 60M policy frequently emits ``<|eos|>`` and then keeps generating. Such
    text can never be encoded as a chat response, so it is dropped here rather
    than surfacing later as a Data Manifest validation failure. The predicate is
    the same string test the chat encoder applies.
    """

    return any(token in text for token in SPECIAL_TOKENS)


def _pick_rejected(
    sampled: _Sampled, tokenizer: NativeTokenizer | None = None
) -> str | None:
    """Choose the most frequent wrong output; ties resolved by first occurrence."""

    wrong = [
        text.strip()
        for text, reward in zip(sampled.outputs, sampled.rewards, strict=True)
        if reward == 0.0 and text.strip() and not _contains_control_token(text.strip())
    ]
    if not wrong:
        return None
    counts = collections.Counter(wrong)
    best = max(counts.values())
    return next(text for text in wrong if counts[text] == best)


def build_outputs(
    sampled: list[_Sampled], *, tokenizer: NativeTokenizer | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dpo_records: list[dict[str, Any]] = []
    grpo_records: list[dict[str, Any]] = []
    bucket = collections.Counter()
    for item in sampled:
        record = item.prompt_record
        successes = sum(1 for reward in item.rewards if reward > 0)
        rate = successes / len(item.rewards)
        if successes == len(item.rewards):
            bucket["always_correct"] += 1
        elif successes == 0:
            bucket["never_correct"] += 1
        else:
            bucket["partially_correct"] += 1

        if 0 < rate < 1:
            grpo_records.append(record)

        rejected = _pick_rejected(item, tokenizer)
        if rejected is None or rejected == record["answer"].strip():
            continue
        dpo_records.append(
            {
                "id": f"{record['id']}-onpolicy",
                "source_id": record["source_id"],
                "template_id": record["template_id"],
                "language": record["language"],
                "prompt": record["prompt"],
                "chosen": record["answer"],
                "rejected": rejected,
                "metadata": {
                    "verifier": record["metadata"]["verifier"],
                    "policy_success_rate": rate,
                },
            }
        )
    stats = {
        "prompts": len(sampled),
        "difficulty_buckets": dict(bucket),
        "dpo_pairs": len(dpo_records),
        "grpo_solvable_prompts": len(grpo_records),
    }
    return dpo_records, grpo_records, stats


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_onpolicy_preference.py",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Native SFT checkpoint directory"
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/grpo-source.jsonl"),
        help="verifiable prompts; must expose metadata.verifier",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tokenizer = NativeTokenizer.from_directory(args.tokenizer)
        model_config = load_native_model_config(args.checkpoint / "model/config.json")
        model = NativeTransformer(model_config)
        model.load(args.checkpoint / "model")
        device = torch.device(args.device)
        model.to_device(device)
        model.set_training(False)

        records = _load_records(args.source)
        sampled = _sample_policy(
            model,
            tokenizer,
            records,
            samples=args.samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            batch_size=args.batch_size,
        )
        dpo_records, grpo_records, stats = build_outputs(sampled, tokenizer=tokenizer)

        dpo_path = args.output_dir / "dpo-onpolicy-source.jsonl"
        grpo_path = args.output_dir / "grpo-solvable-source.jsonl"
        stats_path = args.output_dir / "onpolicy_sampling_stats.json"
        write_jsonl(dpo_path, dpo_records)
        write_jsonl(grpo_path, grpo_records)
        payload = {
            "schema_version": "1.0",
            "dataset_license": DATA_LICENSE,
            "policy_checkpoint": str(args.checkpoint),
            "source": str(args.source),
            "samples_per_prompt": args.samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            **stats,
            "outputs": {"dpo": str(dpo_path), "grpo_solvable": str(grpo_path)},
        }
        stats_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"dpo pairs:    {len(dpo_records)} -> {dpo_path}")
    print(f"grpo prompts: {len(grpo_records)} -> {grpo_path}")
    print(f"stats:        {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
