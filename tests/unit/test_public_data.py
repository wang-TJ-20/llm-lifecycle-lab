from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llm_lifecycle_lab.data import (
    available_public_recipes,
    load_public_source_manifest,
    materialize_public_dataset,
    prepare_dataset,
)
from llm_lifecycle_lab.data.public import PublicDatasetRecipe
from llm_lifecycle_lab.exceptions import DataValidationError


def _recipe(max_records: int = 20) -> PublicDatasetRecipe:
    return PublicDatasetRecipe(
        recipe_id="fixture-v1",
        description="Unit-test public source.",
        repository="owner/dataset",
        revision="a" * 40,
        config="default",
        split="train",
        text_field="story",
        source_id_field="generation_id",
        license="MIT",
        language="en",
        synthetic=True,
        upstream_file="data/train.parquet",
        upstream_file_sha256="b" * 64,
        selection_strategy="source-prefix",
        max_records=max_records,
        min_characters=4,
    )


def _rows(count: int = 20) -> list[dict[str, str]]:
    return [
        {
            "generation_id": f"generation-{index}",
            "story": f"Public story number {index} with enough text.",
        }
        for index in range(count)
    ]


class PublicDatasetTests(unittest.TestCase):
    def test_builtin_recipes_are_nested_and_pinned(self) -> None:
        recipes = {recipe.recipe_id: recipe for recipe in available_public_recipes()}
        smoke = recipes["simplestories-smoke-v1"]
        learn = recipes["simplestories-60m-v1"]
        chinese_smoke = recipes["wikipedia-zh-smoke-v1"]
        chinese_learn = recipes["wikipedia-zh-60m-v1"]

        self.assertEqual(len(recipes), 4)
        self.assertEqual(smoke.repository, "SimpleStories/SimpleStories")
        self.assertEqual(smoke.revision, learn.revision)
        self.assertEqual(smoke.upstream_file, learn.upstream_file)
        self.assertEqual(
            smoke.upstream_file_sha256,
            learn.upstream_file_sha256,
        )
        self.assertEqual(smoke.max_records, 10_000)
        self.assertEqual(learn.max_records, 100_000)
        self.assertEqual(smoke.license, "MIT")
        self.assertEqual(
            smoke.expected_source_sha256,
            "861ca23380e0d6a718350038e74fedd4f7d824bf9e18c3d907bd01fa88be346e",
        )
        self.assertEqual(
            learn.expected_source_sha256,
            "e1ddf88e44e31a0858733c66f9b8ab3eaa7b91db1e73cfb6a97b0a7dd161763e",
        )
        self.assertEqual(chinese_smoke.repository, "wikimedia/wikipedia")
        self.assertEqual(chinese_smoke.revision, chinese_learn.revision)
        self.assertEqual(chinese_smoke.upstream_file, chinese_learn.upstream_file)
        self.assertEqual(chinese_smoke.max_records, 10_000)
        self.assertEqual(chinese_learn.max_records, 126_000)
        self.assertEqual(chinese_smoke.language, "zh")
        self.assertEqual(chinese_smoke.license, "CC-BY-SA-3.0")
        self.assertEqual(
            chinese_smoke.expected_source_sha256,
            "6dd9f3f86fbb3101dffea4315881e25c5516cecaf66bf1afcb442c804bae2739",
        )
        self.assertEqual(
            chinese_learn.expected_source_sha256,
            "fdf06f43db54c564945725e175d5196f93824a1b411558763ff30d318128a9a0",
        )

    def test_materialize_requires_exact_license_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "source"
            with (
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    return_value=_recipe(),
                ),
                self.assertRaisesRegex(DataValidationError, "accept-license MIT"),
            ):
                materialize_public_dataset(
                    "fixture-v1",
                    output,
                    accepted_license="Apache-2.0",
                )

            self.assertFalse(output.exists())

    def test_materialize_rejects_output_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "source"
            recipe = replace(_recipe(max_records=1), expected_source_sha256="c" * 64)
            with (
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    return_value=recipe,
                ),
                patch(
                    "llm_lifecycle_lab.data.public._load_huggingface_rows",
                    return_value=(_rows(1), {"loader": "test-double"}),
                ),
                self.assertRaisesRegex(DataValidationError, "output hash mismatch"),
            ):
                materialize_public_dataset(
                    "fixture-v1",
                    output,
                    accepted_license="MIT",
                )

            self.assertFalse(output.exists())

    def test_materialize_and_prepare_preserve_public_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public_source = root / "public-source"
            with (
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    return_value=_recipe(),
                ),
                patch(
                    "llm_lifecycle_lab.data.public._load_huggingface_rows",
                    return_value=(_rows(25), {"loader": "test-double"}),
                ),
            ):
                source_manifest = materialize_public_dataset(
                    "fixture-v1",
                    public_source,
                    accepted_license="mit",
                )

            source_path = public_source / "source.jsonl"
            records = [
                json.loads(line)
                for line in source_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 20)
            self.assertEqual(records[0]["source_id"], "generation-0")
            self.assertEqual(records[-1]["source_id"], "generation-19")
            self.assertEqual(
                records[0]["source"],
                f"hf://datasets/owner/dataset@{'a' * 40}/train",
            )
            self.assertEqual(source_manifest.records, 20)

            prepared = root / "prepared"
            prepare_dataset(
                source_path,
                prepared,
                dataset_id="fixture-public-v1",
                record_kind="pretrain",
                license_name="MIT",
                group_by="source_id",
                source_metadata=source_manifest.to_dict(),
            )
            prepared_manifest = json.loads(
                (prepared / "data_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                prepared_manifest["source_metadata"]["recipe_id"],
                "fixture-v1",
            )
            self.assertEqual(
                prepared_manifest["source_metadata"]["revision"],
                "a" * 40,
            )

    def test_public_source_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "source"
            with (
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    return_value=_recipe(max_records=1),
                ),
                patch(
                    "llm_lifecycle_lab.data.public._load_huggingface_rows",
                    return_value=(_rows(1), {"loader": "test-double"}),
                ),
            ):
                materialize_public_dataset(
                    "fixture-v1",
                    output,
                    accepted_license="MIT",
                )
            with (output / "source.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"tampered":true}\n')

            with (
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    return_value=_recipe(max_records=1),
                ),
                self.assertRaisesRegex(DataValidationError, "hash mismatch"),
            ):
                load_public_source_manifest(output / "source.jsonl")

    def test_public_manifest_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "source"
            recipe = _recipe(max_records=1)
            with (
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    return_value=recipe,
                ),
                patch(
                    "llm_lifecycle_lab.data.public._load_huggingface_rows",
                    return_value=(_rows(1), {"loader": "test-double"}),
                ),
            ):
                materialize_public_dataset(
                    "fixture-v1",
                    output,
                    accepted_license="MIT",
                )

            manifest_path = output / "source_manifest.json"
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value["revision"] = "c" * 40
            manifest_path.write_text(
                json.dumps(value, ensure_ascii=False),
                encoding="utf-8",
            )

            with (
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    return_value=recipe,
                ),
                self.assertRaisesRegex(
                    DataValidationError,
                    "does not match installed recipe.*revision",
                ),
            ):
                load_public_source_manifest(output / "source.jsonl")


if __name__ == "__main__":
    unittest.main()
