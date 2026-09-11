from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch

from llm_lifecycle_lab.exceptions import ArtifactError, ConfigError, ContractError
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

    def test_generation_rejects_unsupported_padding(self) -> None:
        model = NativeTransformer(micro_config())
        tokens = torch.tensor([[1, 2, 3, 0, 0]])
        for mask in (
            [[1, 1, 1, 0, 0]],
            [[1, 0, 1, 1, 1]],
            [[0, 0, 0, 0, 0]],
        ):
            with (
                self.subTest(mask=mask),
                self.assertRaisesRegex(ContractError, "left.padding"),
            ):
                model.generate(
                    input_ids=tokens,
                    attention_mask=torch.tensor(mask),
                    config=GenerationConfig(max_new_tokens=3),
                )
        self.assertTrue(model.training)

    def test_left_padded_generation_matches_individual_prompts(self) -> None:
        model = NativeTransformer(micro_config())
        tokens = torch.tensor([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        config = GenerationConfig(max_new_tokens=3)
        batched = model.generate(input_ids=tokens, attention_mask=mask, config=config)
        for index, start in ((0, 2), (1, 0)):
            single = model.generate(
                input_ids=tokens[index : index + 1, start:],
                attention_mask=None,
                config=config,
            )
            torch.testing.assert_close(
                batched.token_ids[index, -3:], single.token_ids[0, -3:]
            )
        self.assertTrue(model.training)

    def test_generation_restores_eval_mode(self) -> None:
        model = NativeTransformer(micro_config()).eval()
        model.generate(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=None,
            config=GenerationConfig(max_new_tokens=2),
        )
        self.assertFalse(model.training)

    def test_generation_validates_input_before_changing_mode(self) -> None:
        model = NativeTransformer(micro_config())
        for tokens, mask in (
            (torch.empty((0, 3), dtype=torch.long), None),
            (torch.empty((1, 0), dtype=torch.long), None),
            (torch.tensor([[1, 2, 3]]), torch.ones((1, 2))),
            (torch.tensor([[1, 2, 3]]), torch.tensor([[0.5, 1, 1]])),
        ):
            with (
                self.subTest(tokens=tokens, mask=mask),
                self.assertRaises(ContractError),
            ):
                model.generate(
                    input_ids=tokens,
                    attention_mask=mask,
                    config=GenerationConfig(max_new_tokens=2),
                )
            self.assertTrue(model.training)
        for field in ("eos_token_id", "pad_token_id"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ContractError, field),
            ):
                model.generate(
                    input_ids=torch.tensor([[1, 2, 3]]),
                    attention_mask=None,
                    config=GenerationConfig(max_new_tokens=2, **{field: 64}),
                )

    def test_generation_pads_finished_rows_and_stops_at_eos(self) -> None:
        model = NativeTransformer(micro_config())
        with patch(
            "llm_lifecycle_lab.model.native.generation.select_next_token",
            side_effect=[torch.tensor([2, 4]), torch.tensor([6, 2])],
        ) as select:
            output = model.generate(
                input_ids=torch.tensor([[1, 3], [1, 5]]),
                attention_mask=None,
                config=GenerationConfig(
                    max_new_tokens=4, eos_token_id=2, pad_token_id=0
                ),
            )
        self.assertEqual(select.call_count, 2)
        self.assertEqual(output.token_ids.tolist(), [[1, 3, 2, 0], [1, 5, 4, 2]])
        self.assertEqual(output.stop_reason, "eos")
        self.assertEqual(output.generated_tokens, 2)

    def test_generation_restores_mode_when_sampling_fails(self) -> None:
        model = NativeTransformer(micro_config())
        with (
            patch(
                "llm_lifecycle_lab.model.native.generation.select_next_token",
                side_effect=RuntimeError("sampling failed"),
            ) as select,
            self.assertRaisesRegex(RuntimeError, "sampling failed"),
        ):
            model.generate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=None,
                config=GenerationConfig(max_new_tokens=2),
            )
        select.assert_called_once()
        self.assertTrue(model.training)

    def test_sampling_is_seeded_without_changing_cpu_rng(self) -> None:
        model = NativeTransformer(micro_config())
        config = GenerationConfig(max_new_tokens=4, do_sample=True, top_p=0.8, seed=11)
        state = torch.get_rng_state().clone()
        first = model.generate(
            input_ids=torch.tensor([[1, 2, 3]]), attention_mask=None, config=config
        )
        second = model.generate(
            input_ids=torch.tensor([[1, 2, 3]]), attention_mask=None, config=config
        )
        torch.testing.assert_close(first.token_ids, second.token_ids)
        torch.testing.assert_close(state, torch.get_rng_state())

    def test_unpadded_forward_matches_explicit_mask_and_gradients(self) -> None:
        tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        for qk_norm in (False, True):
            with self.subTest(qk_norm=qk_norm):
                config = replace(micro_config(), qk_norm=qk_norm)
                fast = NativeTransformer(config)
                explicit = NativeTransformer(config)
                explicit.load_state_dict(fast.state_dict())
                fast_logits = fast(input_ids=tokens).logits
                explicit_logits = explicit(
                    input_ids=tokens, attention_mask=torch.ones_like(tokens)
                ).logits
                torch.testing.assert_close(fast_logits, explicit_logits)
                fast_logits.square().mean().backward()
                explicit_logits.square().mean().backward()
                for left, right in zip(
                    fast.parameters(), explicit.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        left.grad, right.grad, rtol=1e-4, atol=1e-6
                    )

    def test_padding_mask_is_shared_by_all_layers(self) -> None:
        model = NativeTransformer(micro_config()).eval()
        observed = []

        def record_mask(_module, _args, kwargs):
            observed.append((kwargs["attention_mask"], kwargs["is_causal"]))

        handles = [
            layer.attention.register_forward_pre_hook(record_mask, with_kwargs=True)
            for layer in model.layers
        ]
        try:
            model(
                input_ids=torch.tensor([[1, 2, 3, 0]]),
                attention_mask=torch.tensor([[1, 1, 1, 0]]),
            )
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(observed), model.config.num_hidden_layers)
        expected = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, False],
            ]
        )[None, None]
        torch.testing.assert_close(observed[0][0], expected)
        self.assertTrue(
            all(mask is observed[0][0] and not causal for mask, causal in observed)
        )

    def test_padded_cache_matches_full_forward(self) -> None:
        tokens = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]])
        mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        for qk_norm in (False, True):
            model = NativeTransformer(replace(micro_config(), qk_norm=qk_norm)).eval()
            with self.subTest(qk_norm=qk_norm), torch.inference_mode():
                full = model(input_ids=tokens, attention_mask=mask).logits
                for split in (3, 5):
                    prefix = model(
                        input_ids=tokens[:, :split],
                        attention_mask=mask[:, :split],
                        use_cache=True,
                    )
                    suffix = model(
                        input_ids=tokens[:, split:],
                        attention_mask=mask,
                        cache=prefix.cache,
                    )
                    torch.testing.assert_close(full[:, split:], suffix.logits)

    def test_last_logits_preserve_outputs_cache_and_gradients(self) -> None:
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        for keep in (1, 3):
            with self.subTest(keep=keep):
                full_model = NativeTransformer(micro_config())
                sliced_model = NativeTransformer(micro_config())
                sliced_model.load_state_dict(full_model.state_dict())
                full = full_model(input_ids=tokens, use_cache=True)
                sliced = sliced_model(
                    input_ids=tokens, use_cache=True, logits_to_keep=keep
                )
                self.assertEqual(tuple(sliced.logits.shape), (1, keep, 64))
                torch.testing.assert_close(full.logits[:, -keep:], sliced.logits)
                for full_kv, sliced_kv in zip(full.cache, sliced.cache, strict=True):
                    torch.testing.assert_close(full_kv, sliced_kv)
                full.logits[:, -keep:].square().mean().backward()
                sliced.logits.square().mean().backward()
                for left, right in zip(
                    full_model.parameters(), sliced_model.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        left.grad, right.grad, rtol=1e-4, atol=1e-6
                    )

    def test_last_logits_validates_count(self) -> None:
        model = NativeTransformer(micro_config())
        for value in (-1, 4, True, 1.5):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ContractError, "logits_to_keep"),
            ):
                model(input_ids=torch.tensor([[1, 2, 3]]), logits_to_keep=value)

    def test_generation_projects_only_one_position_per_step(self) -> None:
        model = NativeTransformer(micro_config())
        shapes = []

        def record_shape(_module, args):
            shapes.append(tuple(args[0].shape))

        handle = model.lm_head.register_forward_pre_hook(record_shape)
        try:
            model.generate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=None,
                config=GenerationConfig(max_new_tokens=3),
            )
        finally:
            handle.remove()
        self.assertEqual(shapes, [(1, 1, 32)] * 3)

    def test_checkpoint_rejects_fractional_dimension(self) -> None:
        model = NativeTransformer(micro_config())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model"
            model.save(checkpoint)
            path = checkpoint / "config.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["num_hidden_layers"] = 2.9
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "num_hidden_layers"):
                model.load(checkpoint)

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
