from __future__ import annotations

import unittest

from llm_lifecycle_lab.exceptions import ConfigError
from llm_lifecycle_lab.training import EngineConfig


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


if __name__ == "__main__":
    unittest.main()
