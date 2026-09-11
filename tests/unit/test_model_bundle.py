from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.contracts import ModelMetadata, ModelRoute
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model import (
    GenerationConfig,
    GenerationOutput,
    ModelBundle,
    ModelOutput,
)


class DummyModel:
    def forward(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None = None,
        use_cache: bool = False,
        cache: Any | None = None,
    ) -> ModelOutput:
        return ModelOutput(logits=input_ids)

    def generate(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None,
        config: GenerationConfig,
    ) -> GenerationOutput:
        return GenerationOutput(
            token_ids=input_ids,
            prompt_tokens=1,
            generated_tokens=0,
            stop_reason="length",
        )

    def trainable_parameters(self) -> list[tuple[str, Any]]:
        return []

    def set_training(self, training: bool) -> None:
        return None

    def is_training(self) -> bool:
        return False

    def to_device(self, device: Any) -> None:
        return None

    def save(self, path: Path) -> None:
        return None

    def load(self, path: Path) -> None:
        return None


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(text)]

    def decode(self, token_ids: list[int]) -> str:
        return str(token_ids)


class ModelBundleTests(unittest.TestCase):
    def test_native_bundle_allows_no_chat_template(self) -> None:
        bundle = ModelBundle(
            model=DummyModel(),
            tokenizer=DummyTokenizer(),
            chat_template=None,
            metadata=ModelMetadata(
                model_route=ModelRoute.NATIVE,
                provider="native",
                model_id="smoke-10m",
                architecture="dense-decoder",
            ),
        )

        self.assertEqual(bundle.metadata.model_id, "smoke-10m")

    def test_qwen_bundle_requires_matching_revisions(self) -> None:
        metadata = ModelMetadata(
            model_route=ModelRoute.QWEN3_TRANSFER,
            provider="huggingface",
            model_id="Qwen/Qwen3-0.6B-Base",
            architecture="qwen3",
            upstream_revision="a" * 40,
            tokenizer_revision="b" * 40,
            chat_template_version="v1",
        )

        with self.assertRaisesRegex(
            ContractError,
            "model and tokenizer revisions must match",
        ):
            ModelBundle(
                model=DummyModel(),
                tokenizer=DummyTokenizer(),
                chat_template="{{ messages }}",
                metadata=metadata,
            )

    def test_bundle_requires_tokenizer_contract(self) -> None:
        with self.assertRaisesRegex(ContractError, "encode"):
            ModelBundle(
                model=DummyModel(),
                tokenizer=object(),
                chat_template=None,
                metadata=ModelMetadata(
                    model_route=ModelRoute.NATIVE,
                    provider="native",
                    model_id="smoke-10m",
                    architecture="dense-decoder",
                ),
            )


if __name__ == "__main__":
    unittest.main()
