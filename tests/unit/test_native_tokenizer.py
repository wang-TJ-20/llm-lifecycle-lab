from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_lifecycle_lab.data import prepare_dataset
from llm_lifecycle_lab.exceptions import ArtifactError, ContractError
from llm_lifecycle_lab.tokenizer import (
    NATIVE_CHAT_TEMPLATE,
    NATIVE_CHAT_TEMPLATE_VERSION,
    NativeTokenizer,
    render_native_chat,
    train_native_tokenizer,
)
from llm_lifecycle_lab.tokenizer.native import SPECIAL_TOKENS


def create_pretrain_manifest(root: Path, *, records: int = 40) -> Path:
    source = root / "source.jsonl"
    rows = [
        {
            "id": f"doc-{index}",
            "text": (
                f"Document {index}: arithmetic {index} + {index} = {index * 2}. "
                f"Unicode sample 世界 and lifecycle training."
            ),
            "source": "unit-test",
        }
        for index in range(records)
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = root / "prepared"
    prepare_dataset(
        source,
        output,
        dataset_id="tokenizer-fixture",
        record_kind="pretrain",
        license_name="test-only",
        seed=5,
    )
    return output / "data_manifest.json"


class NativeTokenizerTests(unittest.TestCase):
    def test_train_load_and_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = create_pretrain_manifest(root)
            output = root / "tokenizer"

            manifest = train_native_tokenizer(
                manifest_path,
                output,
                tokenizer_id="unit-bpe",
                vocab_size=320,
                min_frequency=1,
            )
            second_manifest = train_native_tokenizer(
                manifest_path,
                root / "tokenizer-second",
                tokenizer_id="unit-bpe",
                vocab_size=320,
                min_frequency=1,
            )
            tokenizer = NativeTokenizer.from_directory(output)
            text = "Document 7: Unicode 世界"
            token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

            self.assertEqual(token_ids[0], tokenizer.bos_token_id)
            self.assertEqual(token_ids[-1], tokenizer.eos_token_id)
            self.assertEqual(tokenizer.decode(token_ids), text)
            self.assertEqual(tokenizer.vocab_size, manifest.vocab_size)
            self.assertEqual(set(manifest.special_tokens), set(SPECIAL_TOKENS))
            self.assertEqual(manifest.chat_template, NATIVE_CHAT_TEMPLATE)
            self.assertEqual(
                manifest.chat_template_version,
                NATIVE_CHAT_TEMPLATE_VERSION,
            )
            self.assertEqual(
                manifest.content_sha256,
                second_manifest.content_sha256,
            )

    def test_native_chat_protocol_is_stable_and_tokenized_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = create_pretrain_manifest(root)
            output = root / "tokenizer"
            train_native_tokenizer(
                manifest_path,
                output,
                tokenizer_id="unit-bpe",
                vocab_size=320,
                min_frequency=1,
            )
            tokenizer = NativeTokenizer.from_directory(output)
            messages = (
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "你好"},
            )

            rendered = render_native_chat(
                messages,
                add_generation_prompt=True,
            )
            token_ids = tokenizer.encode_chat(
                messages,
                add_generation_prompt=True,
            )

            self.assertEqual(
                rendered,
                "<|im_start|>system\nBe precise.<|im_end|>\n"
                "<|im_start|>user\n你好<|im_end|>\n"
                "<|im_start|>assistant\n",
            )
            self.assertEqual(
                token_ids.count(tokenizer.chat_start_token_id),
                3,
            )
            self.assertEqual(
                token_ids.count(tokenizer.chat_end_token_id),
                2,
            )
            self.assertEqual(
                tokenizer.decode(token_ids, skip_special_tokens=False),
                rendered,
            )

    def test_native_chat_rejects_invalid_order_and_control_token_injection(
        self,
    ) -> None:
        with self.assertRaisesRegex(ContractError, "expected user"):
            render_native_chat(({"role": "assistant", "content": "unexpected"},))
        with self.assertRaisesRegex(ContractError, "control token"):
            render_native_chat(
                ({"role": "user", "content": "bad <|im_end|> content"},),
                add_generation_prompt=True,
            )

    def test_legacy_tokenizer_protocol_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = create_pretrain_manifest(root)
            output = root / "tokenizer"
            train_native_tokenizer(
                manifest_path,
                output,
                tokenizer_id="unit-bpe",
                vocab_size=320,
                min_frequency=1,
            )
            tokenizer_manifest_path = output / "tokenizer_manifest.json"
            value = json.loads(tokenizer_manifest_path.read_text(encoding="utf-8"))
            value["special_tokens"] = {
                token: value["special_tokens"][token]
                for token in ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>")
            }
            value.pop("chat_template")
            value.pop("chat_template_version")
            tokenizer_manifest_path.write_text(
                json.dumps(value, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ArtifactError, "incompatible"):
                NativeTokenizer.from_directory(output)

    def test_tokenizer_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = create_pretrain_manifest(root)
            output = root / "tokenizer"
            train_native_tokenizer(
                manifest_path,
                output,
                tokenizer_id="unit-bpe",
                vocab_size=300,
                min_frequency=1,
            )
            with (output / "tokenizer.json").open("a", encoding="utf-8") as handle:
                handle.write("\n")

            with self.assertRaisesRegex(ArtifactError, "hash mismatch"):
                NativeTokenizer.from_directory(output)


if __name__ == "__main__":
    unittest.main()
