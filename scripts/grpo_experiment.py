"""离线验证可验证奖励 GRPO、分组优势、KL、rollout 与精确恢复。

Verify tiny RLVR/GRPO mechanics; synthetic rewards are not a capability claim.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from _project_path import add_project_src_to_path

add_project_src_to_path()

from pretrain_experiment import (
    PlannedInterruption,
    assert_state_equal,
    interrupt_after_second_update,
    prepare_experiment,
)
from sft_experiment import prepare_sft_fixture

from llm_lifecycle_lab.config import dump_yaml
from llm_lifecycle_lab.contracts import RunConfig, Stage
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.grpo import rollout_seed
from llm_lifecycle_lab.data.prepare import prepare_dataset
from llm_lifecycle_lab.evaluation.native import NativeEvaluator
from llm_lifecycle_lab.model.native import load_native_model_config
from llm_lifecycle_lab.model.protocol import GenerationConfig
from llm_lifecycle_lab.training.grpo import run_native_grpo
from llm_lifecycle_lab.training.pretrain import run_native_pretraining
from llm_lifecycle_lab.training.sft import run_native_sft

ROOT = Path(__file__).resolve().parents[1]


def sampled_exact_answer(
    evaluator: NativeEvaluator,
    *,
    example_id: str,
    prompt: str,
    group_size: int,
    max_new_tokens: int,
) -> tuple[str, bool]:
    prompt_ids = evaluator.prompt_ids(
        [{"role": "user", "content": prompt}], "native-chat-v1"
    )
    outputs = []
    for index in range(group_size):
        result = evaluator.model.generate(
            input_ids=torch.tensor([prompt_ids]),
            attention_mask=None,
            config=GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                seed=rollout_seed(example_id, index),
                eos_token_id=evaluator.tokenizer.chat_end_token_id,
                pad_token_id=evaluator.tokenizer.pad_token_id,
            ),
        )
        response = result.token_ids[0, len(prompt_ids) :].tolist()
        if response and response[-1] == evaluator.tokenizer.chat_end_token_id:
            response = response[:-1]
        outputs.append(evaluator.tokenizer.decode(response, skip_special_tokens=False))
    counts = Counter(output for output in outputs if output)
    if not counts:
        return "[no generated answer]", False
    answer, count = min(counts.items(), key=lambda item: (item[1], item[0]))
    return answer, count < group_size


def prepare_grpo_fixture(
    root: Path, *, checkpoint: str, tokenizer_dir: str
) -> tuple[Path, int]:
    evaluator = NativeEvaluator(checkpoint, tokenizer_dir=tokenizer_dir)
    rows = []
    variable_groups = 0
    for index in range(40):
        for language, prompt in (
            ("en", f"Give a short label for synthetic item {index}."),
            ("zh", f"给合成项目{index}一个简短标签。"),
        ):
            example_id = f"grpo-{language}-{index}"
            answer, variable = sampled_exact_answer(
                evaluator,
                example_id=example_id,
                prompt=prompt,
                group_size=4,
                max_new_tokens=4,
            )
            variable_groups += int(variable)
            rows.append(
                {
                    "id": example_id,
                    "source_id": f"reward-{index}",
                    "language": language,
                    "prompt": prompt,
                    "answer": answer,
                    "metadata": {"verifier": "exact"},
                }
            )
    source = root / "grpo-source.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    target = root / "grpo-data"
    prepare_dataset(
        source,
        target,
        dataset_id="grpo-mechanism-fixture",
        record_kind="grpo",
        license_name="Apache-2.0",
        group_by="source_id",
        seed=42,
    )
    return target / "data_manifest.json", variable_groups


def run_experiment(root: Path, *, mode: str) -> dict:
    pretrain_config = prepare_experiment(root)
    model_path = Path(pretrain_config.model["config"])
    model = replace(load_native_model_config(model_path), max_sequence_length=512)
    model_path.write_text(dump_yaml(asdict(model)), encoding="utf-8")
    base = run_native_pretraining(pretrain_config, run_id="base", workdir=ROOT)
    sft_config: RunConfig = replace(
        pretrain_config,
        stage=Stage.SFT,
        model={
            "provider": "native",
            "model_id": model.model_id,
            "init_checkpoint": base.result.final_checkpoint,
            "tokenizer": pretrain_config.model["tokenizer"],
        },
        data={
            "manifest": str(prepare_sft_fixture(root)),
            "evaluation_suite": str(ROOT / "configs/evaluation/lifecycle-v1.yaml"),
        },
        training={**pretrain_config.training, "sequence_length": 512},
    )
    sft = run_native_sft(sft_config, run_id="sft", workdir=ROOT)
    grpo_manifest, variable_groups = prepare_grpo_fixture(
        root,
        checkpoint=sft.result.final_checkpoint,
        tokenizer_dir=pretrain_config.model["tokenizer"],
    )
    parent_hash = sha256_file(Path(sft.result.final_checkpoint) / "model/model.pt")
    grpo_config = replace(
        sft_config,
        stage=Stage.GRPO,
        model={**sft_config.model, "init_checkpoint": sft.result.final_checkpoint},
        data={**sft_config.data, "manifest": str(grpo_manifest)},
        training={
            **sft_config.training,
            "gradient_accumulation_steps": 1,
            "group_size": 4,
            "max_new_tokens": 4,
            "temperature": 1.0,
            "top_p": 1.0,
            "clip_epsilon": 0.2,
            "kl_beta": 0.04,
            "advantage_epsilon": 1e-4,
        },
    )
    continuous = run_native_grpo(grpo_config, run_id="grpo", workdir=ROOT)
    metrics = [
        json.loads(line)
        for line in (continuous.artifacts.path / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    result = {
        "grpo_steps": continuous.result.global_step,
        "rollout_tokens_seen": continuous.result.tokens_seen,
        "synthetic_variable_reward_groups": variable_groups,
        "parent_unchanged": (
            sha256_file(Path(sft.result.final_checkpoint) / "model/model.pt")
            == parent_hash
        ),
        "programmatic_rewards_recorded": any(
            "report_reward_mean" in row for row in metrics
        ),
        "latest_rollouts_recorded": (
            continuous.artifacts.path / "latest_rollouts.json"
        ).is_file(),
        "note": "Adaptive synthetic fixture only; no RL capability claim.",
    }
    if mode == "resume":
        try:
            run_native_grpo(
                grpo_config,
                run_id="interrupted",
                workdir=ROOT,
                metric_callback=interrupt_after_second_update,
            )
        except PlannedInterruption:
            pass
        else:
            raise AssertionError("planned GRPO interruption did not occur")
        resumed = run_native_grpo(grpo_config, resume_run="interrupted", workdir=ROOT)
        left = Path(continuous.result.final_checkpoint)
        right = Path(resumed.result.final_checkpoint)
        for name in ("model/model.pt", "optimizer_state.pt"):
            assert_state_equal(
                torch.load(left / name, map_location="cpu", weights_only=True),
                torch.load(right / name, map_location="cpu", weights_only=True),
            )
        assert_state_equal(
            json.loads((left / "trainer_state.json").read_text()),
            json.loads((right / "trainer_state.json").read_text()),
        )
        result["resume_weights_optimizer_scheduler_rng_equal"] = True
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/grpo_experiment.py", description=__doc__
    )
    parser.add_argument("--mode", choices=("train", "resume"), default="train")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)
    with TemporaryDirectory(prefix="llmlab-grpo-") as temporary:
        result = run_experiment(Path(temporary), mode=args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
