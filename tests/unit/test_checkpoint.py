from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
from llm_lifecycle_lab.training import checkpoint as checkpoint_module
from llm_lifecycle_lab.training.batching import DeterministicBatchStream
from llm_lifecycle_lab.training.checkpoint import (
    CheckpointManager,
    TrainerState,
)


class _StubGradScaler:
    def __init__(self, scale: float = 65536.0) -> None:
        self.state = {"scale": scale}

    def state_dict(self) -> dict[str, float]:
        return dict(self.state)

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.state = dict(state)


class CheckpointTests(unittest.TestCase):
    def test_restore_recovers_training_and_batch_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = RunConfig(
                model_route=ModelRoute.NATIVE,
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
                model_route=ModelRoute.NATIVE,
                stage=Stage.PRETRAIN,
                tokenizer_sha256="0" * 64,
                config_sha256=config_sha256(config),
            )
            scaler = _StubGradScaler(scale=1024.0)
            checkpoint = manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stream=stream,
                state=TrainerState(global_step=1, tokens_seen=8),
                scaler=scaler,
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
            restored_scaler = _StubGradScaler(scale=1.0)

            restored_state = manager.load(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stream=stream,
                scaler=restored_scaler,
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
            self.assertEqual(restored_scaler.state_dict()["scale"], 1024.0)

            legacy_state = torch.load(
                checkpoint / "optimizer_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            legacy_state.pop("scaler")
            legacy_state.pop("mps_rng_state")
            torch.save(legacy_state, checkpoint / "optimizer_state.pt")
            legacy_scaler = _StubGradScaler(scale=2.0)
            manager.load(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stream=stream,
                scaler=legacy_scaler,
            )
            self.assertEqual(legacy_scaler.state_dict()["scale"], 2.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA backend")
    def test_real_grad_scaler_state_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = RunConfig(
                model_route=ModelRoute.NATIVE,
                run_profile=RunProfile.SMOKE,
                stage=Stage.PRETRAIN,
                model={"provider": "native", "model_id": "micro"},
            )
            artifacts = ArtifactStore(Path(temporary) / "runs").create_run(
                config,
                run_id="scaler-test",
            )
            model = NativeTransformer(
                NativeModelConfig(
                    model_id="micro",
                    vocab_size=32,
                    num_hidden_layers=1,
                    hidden_size=16,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    intermediate_size=32,
                    max_sequence_length=16,
                )
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            stream = DeterministicBatchStream(
                list(range(10)),
                batch_size=2,
                seed=3,
                collate_fn=lambda values: values,
            )
            manager = CheckpointManager(
                artifacts,
                model_route=ModelRoute.NATIVE,
                stage=Stage.PRETRAIN,
                tokenizer_sha256="0" * 64,
                config_sha256=config_sha256(config),
            )
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            advanced = scaler.state_dict()
            advanced["scale"] = 512.0
            advanced["_growth_tracker"] = 7
            scaler.load_state_dict(advanced)
            checkpoint = manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stream=stream,
                state=TrainerState(global_step=1, tokens_seen=8),
                scaler=scaler,
            )
            expected = scaler.state_dict()
            restored_scaler = torch.amp.GradScaler("cuda", enabled=True)
            manager.load(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stream=stream,
                scaler=restored_scaler,
            )
            self.assertEqual(restored_scaler.state_dict(), expected)

    def test_mps_rng_state_round_trips_when_available(self) -> None:
        sentinel = torch.arange(8, dtype=torch.uint8)
        recorded: dict[str, object] = {}

        def _fake_get() -> torch.Tensor:
            recorded["get"] = True
            return sentinel

        def _fake_set(value: torch.Tensor) -> None:
            recorded["set"] = value

        with (
            mock.patch.object(
                checkpoint_module,
                "_mps_available",
                return_value=True,
            ),
            mock.patch(
                "torch.mps.get_rng_state",
                side_effect=_fake_get,
            ),
            mock.patch(
                "torch.mps.set_rng_state",
                side_effect=_fake_set,
            ),
        ):
            state = checkpoint_module._mps_rng_state()
            self.assertTrue(torch.equal(state, sentinel))
            checkpoint_module._set_mps_rng_state(sentinel)

        self.assertIs(recorded.get("get"), True)
        self.assertTrue(torch.equal(recorded["set"], sentinel))

    def test_mps_rng_helpers_are_inert_without_mps(self) -> None:
        with mock.patch.object(
            checkpoint_module,
            "_mps_available",
            return_value=False,
        ):
            self.assertIsNone(checkpoint_module._mps_rng_state())
            checkpoint_module._set_mps_rng_state(torch.arange(4, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
