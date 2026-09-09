from __future__ import annotations

import unittest

from llm_lifecycle_lab.contracts import (
    DataSplitManifest,
    ModelRoute,
    RunConfig,
    RunProfile,
    Stage,
)
from llm_lifecycle_lab.exceptions import ContractError


class RunConfigTests(unittest.TestCase):
    def test_split_manifest_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(ContractError, "inside"):
            DataSplitManifest(
                name="train",
                path="../outside.jsonl",
                sha256="0" * 64,
                records=1,
                groups=1,
            )

    def test_native_smoke_config_serializes_enum_values(self) -> None:
        config = RunConfig(
            model_route=ModelRoute.NATIVE_SMOKE,
            run_profile=RunProfile.SMOKE,
            stage=Stage.PRETRAIN,
            model={"provider": "native", "model_id": "smoke-10m"},
        )

        self.assertEqual(config.to_dict()["model_route"], "native-smoke")
        self.assertEqual(config.to_dict()["stage"], "pretrain")

    def test_qwen_route_rejects_pretraining(self) -> None:
        with self.assertRaisesRegex(
            ContractError,
            "cannot run tokenizer training or pretraining",
        ):
            RunConfig(
                model_route=ModelRoute.QWEN3_TRANSFER,
                run_profile=RunProfile.LEARN,
                stage=Stage.PRETRAIN,
                model={
                    "provider": "huggingface",
                    "model_id": "Qwen/Qwen3-0.6B-Base",
                    "revision": "abc",
                    "tokenizer_revision": "abc",
                    "transformers_version": "4.51.3",
                },
            )

    def test_qwen_route_rejects_mismatched_tokenizer_revision(self) -> None:
        with self.assertRaisesRegex(
            ContractError,
            "model and tokenizer revisions must match",
        ):
            RunConfig(
                model_route=ModelRoute.QWEN3_TRANSFER,
                run_profile=RunProfile.LEARN,
                stage=Stage.SFT,
                model={
                    "provider": "huggingface",
                    "model_id": "Qwen/Qwen3-0.6B-Base",
                    "revision": "a" * 40,
                    "tokenizer_revision": "b" * 40,
                    "transformers_version": "4.51.3",
                },
            )

    def test_nested_config_is_immutable_after_validation(self) -> None:
        config = RunConfig(
            model_route=ModelRoute.NATIVE_SMOKE,
            run_profile=RunProfile.SMOKE,
            stage=Stage.PRETRAIN,
            model={"provider": "native", "model_id": "smoke-10m"},
            training={"optimizer": {"learning_rate": 0.001}},
        )

        optimizer = config.training["optimizer"]
        with self.assertRaises(TypeError):
            optimizer["learning_rate"] = 0.1

    def test_non_finite_config_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "must be finite"):
            RunConfig(
                model_route=ModelRoute.NATIVE_SMOKE,
                run_profile=RunProfile.SMOKE,
                stage=Stage.PRETRAIN,
                model={"provider": "native", "model_id": "smoke-10m"},
                training={"learning_rate": float("nan")},
            )

    def test_unknown_top_level_config_field_fails(self) -> None:
        with self.assertRaisesRegex(ContractError, "unknown run config fields"):
            RunConfig.from_dict(
                {
                    "model_route": "native-smoke",
                    "run_profile": "smoke",
                    "stage": "pretrain",
                    "model": {
                        "provider": "native",
                        "model_id": "smoke-10m",
                    },
                    "typo": True,
                }
            )

    def test_invalid_seed_type_is_wrapped_as_contract_error(self) -> None:
        with self.assertRaisesRegex(ContractError, "invalid run config value"):
            RunConfig.from_dict(
                {
                    "model_route": "native-smoke",
                    "run_profile": "smoke",
                    "stage": "pretrain",
                    "model": {
                        "provider": "native",
                        "model_id": "smoke-10m",
                    },
                    "seed": None,
                }
            )


if __name__ == "__main__":
    unittest.main()
