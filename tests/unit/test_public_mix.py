from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llm_lifecycle_lab.data.mix import (
    PublicMixtureRecipe,
    available_public_mixture_recipes,
    load_public_mixture_manifest,
    materialize_public_mixture,
)
from llm_lifecycle_lab.data.prepare import prepare_dataset
from llm_lifecycle_lab.data.public import (
    PublicDatasetRecipe,
    materialize_public_dataset,
)
from llm_lifecycle_lab.exceptions import DataValidationError


def _source_recipe(
    recipe_id: str,
    language: str,
    marker: str,
    *,
    max_records: int = 4,
) -> PublicDatasetRecipe:
    return PublicDatasetRecipe(
        recipe_id=recipe_id,
        description=f"{language} fixture.",
        repository=f"owner/{recipe_id}",
        revision=marker * 40,
        config="default",
        split="train",
        text_field="text",
        source_id_field="source_id",
        license="MIT" if language == "en" else "Apache-2.0",
        language=language,
        synthetic=language == "en",
        upstream_file="data/train.parquet",
        upstream_file_sha256=marker * 64,
        selection_strategy="source-prefix",
        max_records=max_records,
        min_characters=4,
    )


def _mixture_recipe(
    expected_source_sha256: str | None = None,
) -> PublicMixtureRecipe:
    return PublicMixtureRecipe(
        mixture_id="bilingual-fixture-v1",
        description="Bilingual fixture.",
        components=("english-fixture-v1", "chinese-fixture-v1"),
        strategy="round-robin",
        expected_source_sha256=expected_source_sha256,
    )


def _rows(language: str, count: int = 4) -> list[dict[str, str]]:
    prefix = "English text" if language == "en" else "中文文本"
    return [
        {
            "source_id": f"{language}-{index}",
            "text": f"{prefix} number {index}.",
        }
        for index in range(count)
    ]


def _materialize_sources(
    root: Path,
    *,
    english_records: int = 4,
    chinese_records: int = 4,
) -> tuple[Path, Path, dict[str, PublicDatasetRecipe]]:
    english = _source_recipe(
        "english-fixture-v1",
        "en",
        "a",
        max_records=english_records,
    )
    chinese = _source_recipe(
        "chinese-fixture-v1",
        "zh",
        "b",
        max_records=chinese_records,
    )
    recipes = {
        english.recipe_id: english,
        chinese.recipe_id: chinese,
    }
    paths = []
    for recipe in (english, chinese):
        output = root / recipe.recipe_id
        with (
            patch(
                "llm_lifecycle_lab.data.public.load_public_recipe",
                return_value=recipe,
            ),
            patch(
                "llm_lifecycle_lab.data.public._load_huggingface_rows",
                return_value=(
                    _rows(recipe.language, recipe.max_records),
                    {"loader": "test-double"},
                ),
            ),
        ):
            materialize_public_dataset(
                recipe.recipe_id,
                output,
                accepted_license=recipe.license,
            )
        paths.append(output / "source.jsonl")
    return paths[0], paths[1], recipes


class PublicMixtureTests(unittest.TestCase):
    def test_builtin_mixtures_pin_language_matched_components(self) -> None:
        mixtures = {
            recipe.mixture_id: recipe for recipe in available_public_mixture_recipes()
        }

        self.assertEqual(len(mixtures), 3)
        self.assertEqual(
            mixtures["bilingual-smoke-v1"].components,
            ("simplestories-smoke-v1", "wikipedia-zh-smoke-v1"),
        )
        self.assertEqual(
            mixtures["bilingual-60m-v1"].components,
            ("simplestories-60m-v1", "wikipedia-zh-60m-v1"),
        )
        self.assertEqual(
            mixtures["bilingual-60m-v2"].components,
            (
                "wikipedia-en-primary-60m-v2",
                "wikipedia-zh-60m-v2",
                "wikipedia-en-secondary-60m-v2",
                "simplestories-60m-v1",
            ),
        )
        self.assertTrue(mixtures["bilingual-60m-v2"].annotate_component)
        self.assertEqual(
            mixtures["bilingual-smoke-v1"].expected_source_sha256,
            "4a6cb466167566d151dd4f597f29522a5157047b93989917776d8983a6118883",
        )
        self.assertEqual(
            mixtures["bilingual-60m-v1"].expected_source_sha256,
            "af07e4c3a610637239a3295fd4754af316bd74c5d08bc25b63fd04a147339589",
        )

    def test_round_robin_mixture_is_order_independent_and_traced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            english_path, chinese_path, recipes = _materialize_sources(root)

            with (
                patch(
                    "llm_lifecycle_lab.data.mix.load_public_mixture_recipe",
                    return_value=_mixture_recipe(),
                ),
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    side_effect=lambda recipe_id: recipes[recipe_id],
                ),
            ):
                manifest = materialize_public_mixture(
                    "bilingual-fixture-v1",
                    [chinese_path, english_path],
                    root / "mixture",
                )

            records = [
                json.loads(line)
                for line in (root / "mixture" / "source.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(manifest.records, 8)
            self.assertEqual(
                [record["language"] for record in records],
                ["en", "zh"] * 4,
            )
            self.assertEqual(
                [component["recipe_id"] for component in manifest.components],
                ["english-fixture-v1", "chinese-fixture-v1"],
            )
            self.assertEqual(manifest.license, "Apache-2.0 AND MIT")

            with (
                patch(
                    "llm_lifecycle_lab.data.mix.load_public_mixture_recipe",
                    return_value=_mixture_recipe(manifest.source_sha256),
                ),
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    side_effect=lambda recipe_id: recipes[recipe_id],
                ),
            ):
                loaded = load_public_mixture_manifest(root / "mixture" / "source.jsonl")
            self.assertEqual(loaded, manifest)

            prepared = root / "prepared"
            prepare_dataset(
                root / "mixture" / "source.jsonl",
                prepared,
                dataset_id="bilingual-fixture-v1",
                record_kind="pretrain",
                license_name="Apache-2.0 AND MIT",
                group_by="source_id",
                source_metadata=loaded.to_dict(),
            )
            data_manifest = json.loads(
                (prepared / "data_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                data_manifest["source_metadata"]["mixture_id"],
                "bilingual-fixture-v1",
            )

    def test_round_robin_preserves_unequal_component_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            english_path, chinese_path, recipes = _materialize_sources(
                root,
                english_records=2,
                chinese_records=4,
            )

            with (
                patch(
                    "llm_lifecycle_lab.data.mix.load_public_mixture_recipe",
                    return_value=_mixture_recipe(),
                ),
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    side_effect=lambda recipe_id: recipes[recipe_id],
                ),
            ):
                manifest = materialize_public_mixture(
                    "bilingual-fixture-v1",
                    [english_path, chinese_path],
                    root / "mixture",
                )

            records = [
                json.loads(line)
                for line in (root / "mixture" / "source.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(manifest.records, 6)
            self.assertEqual(
                [record["language"] for record in records],
                ["en", "zh", "en", "zh", "zh", "zh"],
            )

    def test_mixture_rejects_missing_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            english_path, _, recipes = _materialize_sources(root)

            with (
                patch(
                    "llm_lifecycle_lab.data.mix.load_public_mixture_recipe",
                    return_value=_mixture_recipe(),
                ),
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    side_effect=lambda recipe_id: recipes[recipe_id],
                ),
                self.assertRaisesRegex(DataValidationError, "missing"),
            ):
                materialize_public_mixture(
                    "bilingual-fixture-v1",
                    [english_path],
                    root / "mixture",
                )

    def test_mixture_rejects_output_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            english_path, chinese_path, recipes = _materialize_sources(root)
            recipe = replace(
                _mixture_recipe(),
                expected_source_sha256="c" * 64,
            )

            with (
                patch(
                    "llm_lifecycle_lab.data.mix.load_public_mixture_recipe",
                    return_value=recipe,
                ),
                patch(
                    "llm_lifecycle_lab.data.public.load_public_recipe",
                    side_effect=lambda recipe_id: recipes[recipe_id],
                ),
                self.assertRaisesRegex(DataValidationError, "output hash mismatch"),
            ):
                materialize_public_mixture(
                    "bilingual-fixture-v1",
                    [english_path, chinese_path],
                    root / "mixture",
                )

            self.assertFalse((root / "mixture").exists())


if __name__ == "__main__":
    unittest.main()
