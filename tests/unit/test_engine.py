from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from llm_lifecycle_lab.artifacts import ArtifactStore
from llm_lifecycle_lab.config import config_sha256
from llm_lifecycle_lab.contracts import (
    ModelRoute,
    RunConfig,
    RunProfile,
    Stage,
)
from llm_lifecycle_lab.exceptions import ConfigError
from llm_lifecycle_lab.model.protocol import ModelOutput
from llm_lifecycle_lab.training import EngineConfig
from llm_lifecycle_lab.training.batching import DeterministicBatchStream
from llm_lifecycle_lab.training.checkpoint import CheckpointManager
from llm_lifecycle_lab.training.engine import TrainingEngine
from llm_lifecycle_lab.training.objective import ObjectiveOutput


class EngineConfigTests(unittest.TestCase):
    def test_training_budget_requires_exactly_one_limit(self) -> None:
        with self.assertRaisesRegex(ConfigError, "exactly one"):
            EngineConfig(sequence_length=16)
        with self.assertRaisesRegex(ConfigError, "exactly one"):
            EngineConfig(
                sequence_length=16,
                max_steps=2,
                num_epochs=1,
            )

    def test_step_budget_reports_estimated_coverage(self) -> None:
        config = EngineConfig(
            sequence_length=16,
            max_steps=7,
            micro_batch_size=2,
            gradient_accumulation_steps=5,
        )

        resolved, budget = config.resolve_budget(
            examples_per_epoch=100,
            supervised_tokens_per_epoch=1_000,
        )

        self.assertEqual(resolved.max_steps, 7)
        self.assertEqual(budget.mode, "max_steps")
        self.assertEqual(budget.target_train_tokens, 700)
        self.assertAlmostEqual(budget.estimated_epochs, 0.7)

    def test_token_and_epoch_budgets_resolve_to_optimizer_steps(self) -> None:
        token_config = EngineConfig(
            sequence_length=16,
            max_train_tokens=550,
            micro_batch_size=2,
            gradient_accumulation_steps=5,
        )
        epoch_config = EngineConfig(
            sequence_length=16,
            num_epochs=1.0,
            micro_batch_size=2,
            gradient_accumulation_steps=5,
        )

        resolved_tokens, token_budget = token_config.resolve_budget(
            examples_per_epoch=100,
            supervised_tokens_per_epoch=1_000,
        )
        resolved_epoch, epoch_budget = epoch_config.resolve_budget(
            examples_per_epoch=100,
            supervised_tokens_per_epoch=1_000,
        )

        self.assertEqual(resolved_tokens.max_steps, 6)
        self.assertEqual(token_budget.estimated_train_tokens, 600)
        self.assertEqual(resolved_epoch.max_steps, 10)
        self.assertEqual(epoch_budget.target_train_tokens, 1_000)


class _ScalarModel:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
        self._training = False

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        cache: object | None = None,
    ) -> ModelOutput:
        scalar = self.weight.sum()
        logits = scalar * torch.ones_like(input_ids, dtype=torch.float32)
        return ModelOutput(logits=logits)

    def generate(self, **_kwargs: object) -> object:
        raise NotImplementedError

    def trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (("weight", self.weight),)

    def set_training(self, training: bool) -> None:
        self._training = training

    def is_training(self) -> bool:
        return self._training

    def to_device(self, _device: torch.device) -> None:
        return None

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.weight.detach(), path / "weight.pt")

    def load(self, path: Path) -> None:
        self.weight.data.copy_(torch.load(path / "weight.pt", weights_only=True))


class _TokenCountObjective:
    name = "stub-pretrain"

    def __call__(
        self, model: _ScalarModel, batch: dict[str, object]
    ) -> ObjectiveOutput:
        output = model.forward(input_ids=batch["input_ids"])  # type: ignore[arg-type]
        scalar = output.logits.reshape(-1)[0]
        tokens = int(batch["tokens"])  # type: ignore[arg-type]
        loss = scalar * float(batch["scale"])  # type: ignore[arg-type]
        return ObjectiveOutput(
            loss=loss,
            metrics={"supervised_tokens": float(tokens), "source_bytes": float(tokens)},
        )


class EngineGradientAccumulationTests(unittest.TestCase):
    def test_accumulation_weights_micro_batches_by_effective_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_config = RunConfig(
                model_route=ModelRoute.NATIVE,
                run_profile=RunProfile.SMOKE,
                stage=Stage.PRETRAIN,
                model={"provider": "native", "model_id": "stub"},
            )
            artifacts = ArtifactStore(root / "runs").create_run(
                run_config,
                run_id="accumulation-test",
            )
            config = EngineConfig(
                sequence_length=8,
                max_steps=1,
                micro_batch_size=1,
                gradient_accumulation_steps=2,
                learning_rate=1e-6,
                warmup_steps=0,
                gradient_clipping=1e9,
                dtype="float32",
                device="cpu",
                checkpoint_interval=1,
                eval_interval=1,
                log_interval=1,
            )
            resolved, budget = config.resolve_budget(
                examples_per_epoch=2,
                supervised_tokens_per_epoch=40,
            )
            model = _ScalarModel()
            manager = CheckpointManager(
                artifacts,
                model_route=ModelRoute.NATIVE,
                stage=Stage.PRETRAIN,
                tokenizer_sha256="0" * 64,
                config_sha256=config_sha256(run_config),
            )
            engine = TrainingEngine(
                config=resolved,
                budget=budget,
                artifacts=artifacts,
                checkpoint_manager=manager,
                seed=0,
            )
            batches = [
                {
                    "input_ids": torch.zeros(1, 8, dtype=torch.long),
                    "tokens": 10,
                    "scale": 1.0,
                },
                {
                    "input_ids": torch.zeros(1, 8, dtype=torch.long),
                    "tokens": 30,
                    "scale": 2.0,
                },
            ]
            stream = DeterministicBatchStream(
                batches,
                batch_size=1,
                seed=0,
                collate_fn=lambda values: values[0],
            )
            recorded: list[dict[str, object]] = []
            engine.train(
                bundle=SimpleNamespace(model=model),
                objective=_TokenCountObjective(),
                train_stream=stream,
                evaluation_batches=(),
                metric_callback=recorded.append,
            )

            self.assertEqual(len(recorded), 1)
            self.assertAlmostEqual(recorded[0]["train_loss"], 1.75, places=6)
            self.assertAlmostEqual(
                recorded[0]["gradient_norm"],
                1.75 * math.sqrt(2.0),
                places=5,
            )
            self.assertEqual(recorded[0]["tokens"], 40)


if __name__ == "__main__":
    unittest.main()
