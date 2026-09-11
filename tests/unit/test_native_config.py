from __future__ import annotations

import unittest
from dataclasses import asdict, replace

from llm_lifecycle_lab.exceptions import ConfigError
from llm_lifecycle_lab.model.native.config import NativeModelConfig


class NativeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = NativeModelConfig(
            model_id="config-test",
            vocab_size=64,
            num_hidden_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_sequence_length=32,
        )

    def test_config_roundtrip_preserves_values(self) -> None:
        self.assertEqual(NativeModelConfig.from_dict(asdict(self.config)), self.config)

    def test_config_rejects_nonfinite_hyperparameters(self) -> None:
        for field in ("norm_eps", "rope_theta", "initializer_range"):
            for value in (float("nan"), float("inf"), float("-inf")):
                for loaded in (False, True):
                    with (
                        self.subTest(field=field, value=value, loaded=loaded),
                        self.assertRaisesRegex(ConfigError, field),
                    ):
                        if loaded:
                            NativeModelConfig.from_dict(
                                {**asdict(self.config), field: value}
                            )
                        else:
                            replace(self.config, **{field: value})

    def test_config_rejects_noninteger_dimensions(self) -> None:
        for field in (
            "vocab_size",
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "max_sequence_length",
        ):
            for value in (2.9, 2.0, True, "2", None):
                for loaded in (False, True):
                    with (
                        self.subTest(field=field, value=value, loaded=loaded),
                        self.assertRaisesRegex(ConfigError, field),
                    ):
                        if loaded:
                            NativeModelConfig.from_dict(
                                {**asdict(self.config), field: value}
                            )
                        else:
                            replace(self.config, **{field: value})

    def test_config_rejects_nonpositive_dimensions(self) -> None:
        for value in (0, -1):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ConfigError, "num_hidden_layers"),
            ):
                replace(self.config, num_hidden_layers=value)


if __name__ == "__main__":
    unittest.main()
