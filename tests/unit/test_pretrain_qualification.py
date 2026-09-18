from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm_lifecycle_lab.data.prepare import prepare_dataset
from llm_lifecycle_lab.data.qualification import qualify_pretraining_data


class _Tokenizer:
    manifest = SimpleNamespace(
        content_sha256="a" * 64,
        source_data_sha256="b" * 64,
    )

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(range(len(text.split())))


class PretrainingQualificationTests(unittest.TestCase):
    def test_counts_fixed_tokenizer_tokens_and_enforces_composition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            rows = [
                {
                    "id": f"{language}-{index}",
                    "text": "one two three four five six",
                    "source": f"fixture-{language}",
                    "source_id": f"{language}-{index}",
                    "source_recipe_id": f"{language}-source",
                    "language": language,
                }
                for language in ("en", "zh")
                for index in range(40)
            ]
            source.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
            manifest = prepare_dataset(
                source,
                root / "prepared",
                dataset_id="qualification-fixture",
                record_kind="pretrain",
                license_name="MIT",
                group_by="source_id",
                source_metadata={
                    "components": [
                        {
                            "recipe_id": "en-source",
                            "language": "en",
                            "repository": "owner/en",
                            "revision": "c" * 40,
                            "synthetic": True,
                        },
                        {
                            "recipe_id": "zh-source",
                            "language": "zh",
                            "repository": "owner/zh",
                            "revision": "d" * 40,
                            "synthetic": False,
                        },
                    ]
                },
            )
            with patch(
                "llm_lifecycle_lab.data.qualification.NativeTokenizer.from_directory",
                return_value=_Tokenizer(),
            ):
                report = qualify_pretraining_data(
                    root / "prepared/data_manifest.json",
                    root / "tokenizer",
                    minimum_train_tokens=1,
                    maximum_train_tokens=10_000,
                    minimum_language_fraction=0.35,
                    maximum_language_fraction=0.65,
                    maximum_source_fraction=0.65,
                    maximum_synthetic_fraction=0.65,
                )

            self.assertTrue(report["ok"])
            self.assertEqual(report["train_records"], sum(
                split.records for split in manifest.splits if split.name == "train"
            ))
            self.assertEqual(set(report["languages"]), {"en", "zh"})
            self.assertEqual(set(report["sources"]), {"en-source", "zh-source"})


if __name__ == "__main__":
    unittest.main()
