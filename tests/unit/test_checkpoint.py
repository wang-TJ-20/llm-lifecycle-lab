from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from llm_lifecycle_lab.artifacts import ArtifactStore
from llm_lifecycle_lab.config import config_sha256
from llm_lifecycle_lab.contracts import (
    ModelRoute,
    RunConfig,
    RunProfile,
    Stage,
)
from llm_lifecycle_lab.model.native import NativeModelConfig, NativeTransformer
from llm_lifecycle_lab.training.batching import DeterministicBatchStream
from llm_lifecycle_lab.training.checkpoint import (
    CheckpointManager,
    TrainerState,
)


class CheckpointTests(unittest.TestCase):
    def test_restore_recovers_training_and_batch_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = RunConfig(
                model_route=ModelRoute.NATIVE_SMOKE,
                run_profile=RunProfile.SMOKE,
                stage=Stage.PRETRAIN,
                model={"provider": "native", "model_id": "micro"},
            )
            artifacts = ArtifactStore(Path(temporary) / "runs").create_run(
                config,
                run_id="resume-test",
            )
            model_config = NativeModelConfig(
                model_id="micro",
                vocab_size=32,
                num_hidden_layers=1,
                hidden_size=16,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=32,
                max_sequence_length=16,
            )
            model = NativeTransformer(model_config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda _: 1.0,
            )
            stream = DeterministicBatchStream(
                list(range(10)),
                batch_size=2,
                seed=3,
                collate_fn=lambda values: values,
            )
            stream.next_batch()

            input_ids = torch.randint(0, 32, (1, 5))
            loss = model(input_ids=input_ids).logits.square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            manager = CheckpointManager(
                artifacts,
                model_route=ModelRoute.NATIVE_SMOKE,
                stage=Stage.PRETRAIN,
                tokenizer_sha256="0" * 64,
                config_sha256=config_sha256(config),
            )
            checkpoint = manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stream=stream,
                state=TrainerState(global_step=1, tokens_seen=8),
            )
            expected_parameters = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }
            expected_random = torch.rand(4)

            for parameter in model.parameters():
                parameter.data.zero_()
            stream.next_batch()
            torch.manual_seed(999)

            restored_state = manager.load(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stream=stream,
            )
            actual_random = torch.rand(4)

            self.assertEqual(restored_state.global_step, 1)
            self.assertEqual(restored_state.tokens_seen, 8)
            self.assertEqual(stream.state_dict(), {"epoch": 0, "offset": 2})
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(
                    parameter,
                    expected_parameters[name],
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                actual_random,
                expected_random,
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
