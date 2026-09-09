from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from llm_lifecycle_lab.data import prepare_dataset
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.packing import (
    DiskPackedPretrainingDataset,
    materialize_packed_pretraining_dataset,
    verify_packed_pretraining_manifest,
)
from llm_lifecycle_lab.exceptions import DataValidationError
from llm_lifecycle_lab.tokenizer import (
    NativeTokenizer,
    train_native_tokenizer,
)


class PackedPretrainingTests(unittest.TestCase):
    def test_disk_pack_reconstructs_source_tokens_and_detects_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_manifest_path = _prepare_bilingual_data(root)
            tokenizer_path = root / "tokenizer"
            tokenizer_manifest = train_native_tokenizer(
                data_manifest_path,
                tokenizer_path,
                tokenizer_id="packing-test",
                vocab_size=320,
                min_frequency=1,
            )
            packed_path = root / "packed"
            packed_manifest = materialize_packed_pretraining_dataset(
                data_manifest_path,
                tokenizer_path,
                packed_path,
                sequence_length=16,
            )
            tokenizer = NativeTokenizer.from_directory(tokenizer_path)
            on_disk = DiskPackedPretrainingDataset.from_manifest(
                packed_path / "packed_manifest.json",
                split="train",
                data_manifest_sha256=sha256_file(data_manifest_path),
                tokenizer_sha256=tokenizer_manifest.content_sha256,
                sequence_length=16,
            )

            train_split = next(
                split for split in packed_manifest.splits if split.name == "train"
            )
            self.assertEqual(on_disk.stats.documents, train_split.documents)
            self.assertEqual(
                on_disk.stats.supervised_tokens,
                train_split.supervised_tokens,
            )
            self.assertEqual(
                packed_manifest.tokenizer_sha256,
                tokenizer_manifest.content_sha256,
            )

            reconstructed: list[int] = []
            counted_bytes = 0.0
            languages: set[int] = set()
            for index in range(len(on_disk)):
                example = on_disk[index]
                valid_length = int(example["attention_mask"].sum())
                tokens = example["input_ids"][:valid_length].tolist()
                labels = example["labels"][:valid_length].tolist()
                self.assertEqual(tokens, labels)
                self.assertTrue(bool((example["labels"][valid_length:] == -100).all()))
                if reconstructed:
                    self.assertEqual(reconstructed[-1], tokens[0])
                    reconstructed.extend(tokens[1:])
                else:
                    reconstructed.extend(tokens)
                counted_bytes += float(example["source_bytes"])
                languages.update(
                    int(value)
                    for value in example["language_ids"][:valid_length]
                    if int(value) >= 0
                )

            expected_tokens: list[int] = []
            data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
            train_path = data_manifest_path.parent / next(
                split["path"]
                for split in data_manifest["splits"]
                if split["name"] == "train"
            )
            for line in train_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                expected_tokens.extend(
                    [
                        tokenizer.bos_token_id,
                        *tokenizer.encode(record["text"]),
                        tokenizer.eos_token_id,
                    ]
                )
            self.assertEqual(reconstructed, expected_tokens)
            self.assertAlmostEqual(
                counted_bytes,
                train_split.source_bytes,
                places=2,
            )
            self.assertEqual(languages, {0, 1})

            token_path = packed_path / packed_manifest.splits[0].token_ids.path
            external_token_path = root / "external.tokens.i32"
            shutil.copyfile(token_path, external_token_path)
            with token_path.open("ab") as handle:
                handle.write(b"tamper")
            failures = verify_packed_pretraining_manifest(
                packed_path / "packed_manifest.json"
            )
            self.assertTrue(any("size mismatch" in item for item in failures))
            self.assertTrue(any("hash mismatch" in item for item in failures))

            token_path.unlink()
            token_path.symlink_to(external_token_path)
            failures = verify_packed_pretraining_manifest(
                packed_path / "packed_manifest.json"
            )
            self.assertTrue(
                any("regular file inside its directory" in item for item in failures)
            )

    def test_disk_pack_rejects_incompatible_tokenizer_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_manifest_path = _prepare_bilingual_data(root)
            tokenizer_path = root / "tokenizer"
            train_native_tokenizer(
                data_manifest_path,
                tokenizer_path,
                tokenizer_id="packing-test",
                vocab_size=320,
                min_frequency=1,
            )
            packed_path = root / "packed"
            materialize_packed_pretraining_dataset(
                data_manifest_path,
                tokenizer_path,
                packed_path,
                sequence_length=16,
            )

            with self.assertRaisesRegex(DataValidationError, "tokenizer"):
                DiskPackedPretrainingDataset.from_manifest(
                    packed_path / "packed_manifest.json",
                    split="train",
                    data_manifest_sha256=sha256_file(data_manifest_path),
                    tokenizer_sha256="0" * 64,
                    sequence_length=16,
                )


def _prepare_bilingual_data(root: Path) -> Path:
    source = root / "source.jsonl"
    records = [
        {
            "id": f"doc-{index}",
            "text": (
                f"Training sample {index}. 中英文 packing validation document {index}."
            ),
            "source": "unit-test",
            "language": "en" if index % 2 == 0 else "zh",
        }
        for index in range(40)
    ]
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    output = root / "prepared"
    prepare_dataset(
        source,
        output,
        dataset_id="packing-fixture",
        record_kind="pretrain",
        license_name="test-only",
        seed=5,
    )
    return output / "data_manifest.json"


if __name__ == "__main__":
    unittest.main()
