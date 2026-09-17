from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llm_lifecycle_lab.contracts import RecordKind
from llm_lifecycle_lab.data.posttraining import (
    PublicPosttrainingBundleManifest,
    PublicPosttrainingRecipe,
    _load_posttraining_records,
    _transform_helpsteer,
    _transform_msvamp,
    _transform_oasst,
    available_public_posttraining_recipes,
    load_public_posttraining_source_manifest,
    materialize_public_posttraining,
)
from llm_lifecycle_lab.exceptions import DataValidationError


def _small_recipe() -> PublicPosttrainingRecipe:
    base = available_public_posttraining_recipes()[0]
    selection = {
        "oasst1": {
            **dict(base.selection["oasst1"]),
            "records_per_language": 1,
        },
        "helpsteer3": {
            **dict(base.selection["helpsteer3"]),
            "records_per_language": 1,
        },
        "msvamp": {
            **dict(base.selection["msvamp"]),
            "expected_groups": 2,
            "sft_warmup_groups": 1,
        },
    }
    return replace(
        base,
        recipe_id="fixture-public-v1",
        selection=selection,
        expected_source_sha256={"sft": None, "dpo": None, "grpo": None},
    )


def _oasst_rows() -> list[dict]:
    rows = []
    for language in ("en", "zh"):
        tree_id = f"tree-{language}"
        rows.extend(
            (
                {
                    "message_id": f"prompt-{language}",
                    "parent_id": None,
                    "text": "Question" if language == "en" else "问题",
                    "role": "prompter",
                    "lang": language,
                    "review_result": True,
                    "deleted": False,
                    "rank": None,
                    "synthetic": False,
                    "message_tree_id": tree_id,
                    "tree_state": "ready_for_export",
                },
                {
                    "message_id": f"answer-{language}",
                    "parent_id": f"prompt-{language}",
                    "text": "Answer" if language == "en" else "回答",
                    "role": "assistant",
                    "lang": language,
                    "review_result": True,
                    "deleted": False,
                    "rank": 0,
                    "synthetic": False,
                    "message_tree_id": tree_id,
                    "tree_state": "ready_for_export",
                },
            )
        )
    return rows


def _helpsteer_rows() -> list[dict]:
    return [
        {
            "domain": "general",
            "language": "english",
            "context": [{"role": "user", "content": "Question"}],
            "response1": "Chosen one",
            "response2": "Rejected one",
            "overall_preference": -2,
        },
        {
            "domain": "multilingual",
            "language": "chinese",
            "context": [{"role": "user", "content": "问题"}],
            "response1": "较差回答",
            "response2": "较好回答",
            "overall_preference": 3,
        },
    ]


def _msvamp_rows() -> list[dict]:
    return [
        {
            "query": "What is 3 plus 4?",
            "m_query": "三加四等于多少？",
            "equation": "+ number0 number1",
            "response": "7.0",
        },
        {
            "query": "What is 9 minus 2?",
            "m_query": "九减二等于多少？",
            "equation": "- number0 number1",
            "response": "7.0",
        },
    ]


class PublicPosttrainingTests(unittest.TestCase):
    def test_builtin_recipe_is_fully_pinned(self) -> None:
        recipes = available_public_posttraining_recipes()
        self.assertEqual(len(recipes), 1)
        recipe = recipes[0]
        self.assertEqual(recipe.recipe_id, "public-60m-v1")
        self.assertEqual(
            tuple(source.source_id for source in recipe.sources),
            ("oasst1", "helpsteer3", "msvamp"),
        )
        self.assertTrue(all(len(source.revision) == 40 for source in recipe.sources))
        self.assertTrue(
            all(len(source.upstream_file_sha256) == 64 for source in recipe.sources)
        )
        self.assertEqual(
            recipe.expected_source_sha256,
            {
                "sft": (
                    "4286e08a52910b027c2c3d0b7ab1b0333faec11792d918d6a3cc7c483be8110f"
                ),
                "dpo": (
                    "d6a284296bb5b1bfaab2173a28b2d52ffb5669522e8987751a0afd33175f2b0e"
                ),
                "grpo": (
                    "3cc66aa262967b18dee023cbe97336dd875c9db3b995015dedb629fc45abb22b"
                ),
            },
        )

    def test_transforms_preserve_preference_and_partition_semantics(self) -> None:
        recipe = _small_recipe()
        sft, skipped = _transform_oasst(_oasst_rows(), recipe)
        self.assertEqual(skipped, 2)
        self.assertEqual({row["language"] for row in sft}, {"en", "zh"})
        self.assertTrue(all(row["messages"][-1]["role"] == "assistant" for row in sft))

        dpo, skipped = _transform_helpsteer(_helpsteer_rows(), recipe)
        self.assertEqual(skipped, 0)
        by_language = {row["language"]: row for row in dpo}
        self.assertEqual(by_language["en"]["chosen"], "Chosen one")
        self.assertEqual(by_language["zh"]["chosen"], "较好回答")

        warmup, grpo = _transform_msvamp(_msvamp_rows(), recipe)
        self.assertEqual(len(warmup), 2)
        self.assertEqual(len(grpo), 2)
        self.assertFalse(
            {row["source_id"] for row in warmup} & {row["source_id"] for row in grpo}
        )
        self.assertEqual({row["messages"][-1]["content"] for row in warmup}, {"7"})
        self.assertEqual({row["metadata"]["verifier"] for row in grpo}, {"integer"})

    def test_materialize_requires_all_and_only_declared_licenses(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "llm_lifecycle_lab.data.posttraining.load_public_posttraining_recipe",
                return_value=_small_recipe(),
            ),
            self.assertRaisesRegex(DataValidationError, "requires exactly"),
        ):
            materialize_public_posttraining(
                "fixture-public-v1",
                Path(temporary) / "output",
                accepted_licenses=["Apache-2.0"],
            )

    def test_bundle_manifest_rejects_schema_and_stage_drift(self) -> None:
        stages = tuple(
            {
                "record_kind": stage,
                "manifest": f"{stage}/source_manifest.json",
                "records": 1,
                "source_sha256": "0" * 64,
            }
            for stage in ("sft", "dpo", "grpo")
        )
        bundle = PublicPosttrainingBundleManifest(
            manifest_type="public-posttraining-bundle",
            recipe_id="fixture-public-v1",
            stages=stages,
        )
        self.assertEqual(
            [item["record_kind"] for item in bundle.stages],
            ["sft", "dpo", "grpo"],
        )
        with self.assertRaisesRegex(DataValidationError, "schema_version"):
            replace(bundle, schema_version="2.0")
        with self.assertRaisesRegex(DataValidationError, "ordered"):
            replace(bundle, stages=tuple(reversed(stages)))

    def test_materialize_and_verify_each_stage_manifest(self) -> None:
        recipe = _small_recipe()
        oasst, _ = _transform_oasst(_oasst_rows(), recipe)
        dpo, _ = _transform_helpsteer(_helpsteer_rows(), recipe)
        warmup, grpo = _transform_msvamp(_msvamp_rows(), recipe)
        records = {
            RecordKind.SFT: sorted((*oasst, *warmup), key=lambda row: row["id"]),
            RecordKind.DPO: sorted(dpo, key=lambda row: row["id"]),
            RecordKind.GRPO: sorted(grpo, key=lambda row: row["id"]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with (
                patch(
                    "llm_lifecycle_lab.data.posttraining.load_public_posttraining_recipe",
                    return_value=recipe,
                ),
                patch(
                    "llm_lifecycle_lab.data.posttraining._load_posttraining_records",
                    return_value=(records, {"loader": "test-double"}),
                ),
            ):
                bundle = materialize_public_posttraining(
                    "fixture-public-v1",
                    output,
                    accepted_licenses=["CC-BY-4.0", "Apache-2.0"],
                )
                self.assertEqual(len(bundle.stages), 3)
                for stage, expected in (("sft", 4), ("dpo", 2), ("grpo", 2)):
                    manifest = load_public_posttraining_source_manifest(
                        output / stage / "source.jsonl"
                    )
                    self.assertIsNotNone(manifest)
                    assert manifest is not None
                    self.assertEqual(manifest.records, expected)

                source = output / "dpo/source.jsonl"
                with source.open("a", encoding="utf-8") as handle:
                    handle.write('{"tampered":true}\n')
                with self.assertRaisesRegex(DataValidationError, "hash mismatch"):
                    load_public_posttraining_source_manifest(source)

    def test_real_loader_composition_is_stage_complete(self) -> None:
        recipe = _small_recipe()
        oasst, _ = _transform_oasst(_oasst_rows(), recipe)
        dpo, _ = _transform_helpsteer(_helpsteer_rows(), recipe)
        warmup, grpo = _transform_msvamp(_msvamp_rows(), recipe)
        with patch("llm_lifecycle_lab.data.posttraining._download_sources") as download:
            download.return_value = (
                {
                    "oasst1": Path("oasst.parquet"),
                    "helpsteer3": Path("helpsteer.jsonl.gz"),
                    "msvamp": Path("msvamp.jsonl"),
                },
                {},
            )
            with (
                patch(
                    "llm_lifecycle_lab.data.posttraining._read_oasst",
                    return_value=_oasst_rows(),
                ),
                patch(
                    "llm_lifecycle_lab.data.posttraining._read_jsonl_gzip",
                    return_value=_helpsteer_rows(),
                ),
                patch(
                    "llm_lifecycle_lab.data.posttraining._read_jsonl",
                    return_value=_msvamp_rows(),
                ),
            ):
                actual, _ = _load_posttraining_records(recipe)
        self.assertEqual(len(actual[RecordKind.SFT]), len(oasst) + len(warmup))
        self.assertEqual(len(actual[RecordKind.DPO]), len(dpo))
        self.assertEqual(len(actual[RecordKind.GRPO]), len(grpo))


if __name__ == "__main__":
    unittest.main()
