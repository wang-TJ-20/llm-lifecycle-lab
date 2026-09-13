"""不下载 Qwen 权重，验证微型 Qwen3 架构的 LoRA、冻结与精确恢复。

Test a randomly initialized tiny Qwen3/LoRA fixture, never a pretrained Qwen claim.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from _project_path import add_project_src_to_path

add_project_src_to_path()

from pretrain_experiment import (
    PlannedInterruption,
    assert_state_equal,
    interrupt_after_second_update,
)

from llm_lifecycle_lab.artifacts import ArtifactStore
from llm_lifecycle_lab.config import config_sha256
from llm_lifecycle_lab.contracts import ModelRoute, RunConfig, RunProfile, Stage
from llm_lifecycle_lab.interop.assets import require_hf
from llm_lifecycle_lab.interop.backend import HFModelAdapter, apply_lora
from llm_lifecycle_lab.training.batching import DeterministicBatchStream
from llm_lifecycle_lab.training.checkpoint import CheckpointManager
from llm_lifecycle_lab.training.engine import EngineConfig, TrainingEngine
from llm_lifecycle_lab.training.sft import SFTObjective


def run_experiment(root: Path, *, resume: bool) -> dict:
    hf = require_hf(peft=True)
    torch.manual_seed(42)
    base = hf.Qwen3ForCausalLM(
        hf.Qwen3Config(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            tie_word_embeddings=True,
        )
    )
    base.save_pretrained(root / "base", safe_serialization=True)
    base_weights = {
        name: value.detach().clone() for name, value in base.named_parameters()
    }
    settings = {
        "rank": 2,
        "alpha": 4,
        "dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
    }
    config = RunConfig(
        model_route=ModelRoute.NATIVE,
        run_profile=RunProfile.SMOKE,
        stage=Stage.SFT,
        model={"provider": "native", "model_id": "self-generated-qwen3-fixture"},
        output_dir=str(root / "runs"),
        training={"fixture": True},
    )

    # This synthetic engine fixture is not the Qwen transfer route: it has no
    # upstream revision, real tokenizer, external training data, or capability score.
    def train(run_id: str, *, restoring: bool = False, callback=None):
        torch.manual_seed(42)
        model = hf.AutoModelForCausalLM.from_pretrained(
            root / "base",
            local_files_only=True,
            attn_implementation="sdpa",
        )
        adapter = HFModelAdapter(
            apply_lora(model, settings),
            binding={
                "training_method": "lora",
                "fixture": "random-qwen3",
                "lora": settings,
            },
        )
        store = ArtifactStore(config.output_dir)
        artifacts = (
            store.open_run(config, run_id=run_id)
            if restoring
            else (store.create_run(config, run_id=run_id))
        )
        engine_config, budget = EngineConfig(
            sequence_length=8,
            max_steps=3,
            learning_rate=0.01,
            device="cpu",
            checkpoint_interval=1,
            eval_interval=1,
            eval_batches=1,
        ).resolve_budget(examples_per_epoch=2, supervised_tokens_per_epoch=8)
        batches = [
            {
                "input_ids": torch.tensor([[1, 5, 6, 7, 8, 9, 10, 2]]),
                "labels": torch.tensor([[-100, -100, -100, -100, 8, 9, 10, 2]]),
            },
            {
                "input_ids": torch.tensor([[1, 11, 12, 13, 14, 15, 16, 2]]),
                "labels": torch.tensor([[-100, -100, -100, -100, 14, 15, 16, 2]]),
            },
        ]
        manager = CheckpointManager(
            artifacts,
            model_route=config.model_route,
            stage=Stage.SFT,
            tokenizer_sha256="0" * 64,
            config_sha256=config_sha256(config),
        )
        engine = TrainingEngine(
            config=engine_config,
            budget=budget,
            artifacts=artifacts,
            checkpoint_manager=manager,
            seed=42,
        )
        result = engine.train(
            bundle=SimpleNamespace(model=adapter),
            objective=SFTObjective(),
            train_stream=DeterministicBatchStream(
                batches,
                batch_size=1,
                seed=42,
                collate_fn=lambda items: items[0],
            ),
            evaluation_batches=batches,
            resume_from="step-00000001" if restoring else None,
            metric_callback=callback,
        )
        for name, parameter in adapter.raw.get_base_model().named_parameters():
            if "lora_" not in name:
                original_name = name.replace(".base_layer", "")
                torch.testing.assert_close(
                    parameter, base_weights[original_name], rtol=0, atol=0
                )
        assert any(
            torch.count_nonzero(parameter) > 0
            for name, parameter in adapter.trainable_parameters()
            if "lora_B" in name
        )
        return result, adapter

    result, adapter = train("continuous")
    summary = {
        "fixture": "random miniature Qwen3, not Qwen/Qwen3-0.6B-Base weights",
        "steps": result.global_step,
        "base_parameters_unchanged": True,
        "lora_updated": True,
        "trainable_parameters": sum(
            p.numel() for _, p in adapter.trainable_parameters()
        ),
        "total_parameters": sum(p.numel() for p in adapter.raw.parameters()),
    }
    if resume:
        try:
            train("interrupted", callback=interrupt_after_second_update)
        except PlannedInterruption:
            pass
        else:
            raise AssertionError("fixture interruption did not occur")
        recovered, restored = train("interrupted", restoring=True)
        assert_state_equal(adapter.raw.state_dict(), restored.raw.state_dict())
        for name in ("optimizer_state.pt",):
            assert_state_equal(
                torch.load(Path(result.final_checkpoint) / name, weights_only=True),
                torch.load(Path(recovered.final_checkpoint) / name, weights_only=True),
            )
        assert_state_equal(
            json.loads(
                (Path(result.final_checkpoint) / "trainer_state.json").read_text()
            ),
            json.loads(
                (Path(recovered.final_checkpoint) / "trainer_state.json").read_text()
            ),
        )
        summary["resume_weights_optimizer_scheduler_rng_equal"] = True
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/transfer_experiment.py", description=__doc__
    )
    parser.add_argument("--mode", choices=("train", "resume"), default="resume")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)
    with TemporaryDirectory(prefix="llmlab-qwen-fixture-") as temporary:
        report = run_experiment(Path(temporary), resume=args.mode == "resume")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
