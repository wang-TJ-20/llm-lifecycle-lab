from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from llm_lifecycle_lab.exceptions import ConfigError
from llm_lifecycle_lab.model.native import (
    NativeModelConfig,
    NativeTransformer,
    load_native_model_config,
)
from llm_lifecycle_lab.model.native.attention import RotaryEmbedding
from llm_lifecycle_lab.model.protocol import GenerationConfig
from llm_lifecycle_lab.training.stages import causal_lm_loss


def micro_config() -> NativeModelConfig:
    return NativeModelConfig(
        model_id="micro",
        vocab_size=64,
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_sequence_length=32,
    )


class NativeModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_forward_backward_and_weight_tying(self) -> None:
        model = NativeTransformer(micro_config())
        input_ids = torch.randint(0, 64, (2, 12))
        labels = input_ids.clone()

        output = model(input_ids=input_ids)
        loss, supervised_tokens = causal_lm_loss(output.logits, labels)
        loss.backward()

        self.assertEqual(tuple(output.logits.shape), (2, 12, 64))
        self.assertEqual(supervised_tokens, 22)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for _, parameter in model.trainable_parameters()
                if parameter.grad is not None
            )
        )

    def test_causal_mask_blocks_future_tokens(self) -> None:
        model = NativeTransformer(micro_config()).eval()
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 9, 10]])

        with torch.inference_mode():
            first_logits = model(input_ids=first).logits
            second_logits = model(input_ids=second).logits

        torch.testing.assert_close(
            first_logits[:, :3],
            second_logits[:, :3],
            rtol=0,
            atol=1e-6,
        )

    def test_kv_cache_matches_full_forward(self) -> None:
        model = NativeTransformer(micro_config()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        with torch.inference_mode():
            full = model(input_ids=input_ids).logits
            prefix = model(input_ids=input_ids[:, :4], use_cache=True)
            continuation = model(
                input_ids=input_ids[:, 4:],
                attention_mask=torch.ones((1, 6), dtype=torch.bool),
                use_cache=True,
                cache=prefix.cache,
            ).logits

        torch.testing.assert_close(
            continuation,
            full[:, 4:],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_rope_cache_is_shared_and_slices_match_full_sequence(self) -> None:
        model = NativeTransformer(micro_config())

        rotary_modules = [
            module for module in model.modules() if isinstance(module, RotaryEmbedding)
        ]
        self.assertEqual(len(rotary_modules), 1)
        self.assertFalse(any(key.startswith("rotary.") for key in model.state_dict()))

        full_cosine, full_sine = model.rotary(
            start_position=0,
            sequence_length=6,
        )
        prefix_cosine, prefix_sine = model.rotary(
            start_position=0,
            sequence_length=4,
        )
        suffix_cosine, suffix_sine = model.rotary(
            start_position=4,
            sequence_length=2,
        )

        torch.testing.assert_close(
            torch.cat((prefix_cosine, suffix_cosine), dim=2),
            full_cosine,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            torch.cat((prefix_sine, suffix_sine), dim=2),
            full_sine,
            rtol=0,
            atol=0,
        )

    def test_qk_norm_is_optional_and_counted(self) -> None:
        base_config = micro_config()
        qk_config = replace(base_config, qk_norm=True)
        base = NativeTransformer(base_config)
        qk_model = NativeTransformer(qk_config)

        expected_increment = qk_config.num_hidden_layers * 2 * qk_config.head_dim
        self.assertEqual(
            qk_model.parameter_count - base.parameter_count,
            expected_increment,
        )
        self.assertEqual(
            qk_model.parameter_count,
            qk_config.estimated_parameter_count,
        )
        self.assertTrue(
            all(layer.attention.q_norm is not None for layer in qk_model.layers)
        )
        self.assertTrue(
            all(layer.attention.k_norm is not None for layer in qk_model.layers)
        )
        output = qk_model(input_ids=torch.randint(0, 64, (2, 12)))
        self.assertTrue(torch.isfinite(output.logits).all())
        qk_bfloat16 = NativeTransformer(qk_config).to(dtype=torch.bfloat16)
        bfloat16_output = qk_bfloat16(input_ids=torch.randint(0, 64, (2, 12)))
        self.assertEqual(bfloat16_output.logits.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(bfloat16_output.logits).all())
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            autocast_output = qk_model(input_ids=torch.randint(0, 64, (2, 12)))
        self.assertEqual(autocast_output.logits.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(autocast_output.logits).all())

    def test_save_and_load_preserve_logits(self) -> None:
        model = NativeTransformer(micro_config()).eval()
        input_ids = torch.tensor([[2, 4, 6, 8]])
        with torch.inference_mode():
            expected = model(input_ids=input_ids).logits

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model"
            model.save(checkpoint)
            restored = NativeTransformer(micro_config()).eval()
            restored.load(checkpoint)
            with torch.inference_mode():
                actual = restored(input_ids=input_ids).logits

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_legacy_checkpoint_config_defaults_qk_norm_to_false(self) -> None:
        model = NativeTransformer(micro_config()).eval()

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model"
            model.save(checkpoint)
            config_path = checkpoint / "config.json"
            value = json.loads(config_path.read_text(encoding="utf-8"))
            value.pop("qk_norm")
            config_path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            restored = NativeTransformer(micro_config()).eval()
            restored.load(checkpoint)

        for expected, actual in zip(
            model.parameters(),
            restored.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_qk_norm_requires_a_boolean(self) -> None:
        value = {
            **asdict(micro_config()),
            "qk_norm": "false",
        }
        with self.assertRaisesRegex(ConfigError, "qk_norm"):
            NativeModelConfig.from_dict(value)

    def test_greedy_generation_is_deterministic(self) -> None:
        model = NativeTransformer(micro_config())
        prompt = torch.tensor([[1, 2, 3]])
        generation_config = GenerationConfig(max_new_tokens=3, seed=9)

        first = model.generate(
            input_ids=prompt,
            attention_mask=None,
            config=generation_config,
        )
        second = model.generate(
            input_ids=prompt,
            attention_mask=None,
            config=generation_config,
        )

        self.assertTrue(torch.equal(first.token_ids, second.token_ids))
        self.assertEqual(first.generated_tokens, 3)

    def test_profile_parameter_counts_match_names(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        smoke = load_native_model_config(project_root / "configs/models/smoke-10m.yaml")
        learn = load_native_model_config(project_root / "configs/models/tiny-60m.yaml")
        qk_norm = load_native_model_config(
            project_root / "configs/models/tiny-60m-qk-norm.yaml"
        )
        deep_narrow = load_native_model_config(
            project_root / "configs/models/tiny-60m-deep-narrow.yaml"
        )
        deep_narrow_qk = load_native_model_config(
            project_root / "configs/models/tiny-60m-deep-narrow-qk-norm.yaml"
        )

        self.assertGreaterEqual(smoke.estimated_parameter_count, 9_000_000)
        self.assertLess(smoke.estimated_parameter_count, 11_000_000)
        self.assertGreaterEqual(learn.estimated_parameter_count, 55_000_000)
        self.assertLess(learn.estimated_parameter_count, 70_000_000)
        self.assertGreater(
            smoke.token_parameter_share_for_vocab_size(smoke.vocab_size),
            0.5,
        )
        self.assertLess(
            learn.token_parameter_share_for_vocab_size(learn.vocab_size),
            0.21,
        )
        self.assertEqual(
            smoke.parameter_count_for_vocab_size(8_192),
            smoke.estimated_parameter_count - 8_192 * smoke.hidden_size,
        )
        self.assertEqual(
            qk_norm.estimated_parameter_count - learn.estimated_parameter_count,
            learn.num_hidden_layers * 2 * learn.head_dim,
        )
        self.assertLess(
            abs(deep_narrow.estimated_parameter_count - learn.estimated_parameter_count)
            / learn.estimated_parameter_count,
            0.01,
        )
        self.assertEqual(
            deep_narrow_qk.estimated_parameter_count
            - deep_narrow.estimated_parameter_count,
            deep_narrow.num_hidden_layers * 2 * deep_narrow.head_dim,
        )


if __name__ == "__main__":
    unittest.main()
