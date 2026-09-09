from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_lifecycle_lab.config import config_sha256, load_run_config
from llm_lifecycle_lab.exceptions import ConfigError


class ConfigTests(unittest.TestCase):
    def test_repository_pipelines_are_valid_and_stably_hashed(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        paths = sorted((project_root / "configs/pipelines").glob("*.yaml"))

        for path in paths:
            with self.subTest(path=path):
                config = load_run_config(path)
                self.assertEqual(config_sha256(config), config_sha256(config))

    def test_native_60m_ablations_change_only_model_variant(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        pipeline_root = project_root / "configs/pipelines"
        baseline = load_run_config(pipeline_root / "native-v1.yaml").to_dict()
        variants = (
            "native-60m-qk-norm.yaml",
            "native-60m-deep-narrow.yaml",
            "native-60m-deep-narrow-qk-norm.yaml",
        )

        for name in variants:
            with self.subTest(name=name):
                variant = load_run_config(pipeline_root / name).to_dict()
                baseline_model = dict(baseline["model"])
                variant_model = dict(variant["model"])
                baseline_model.pop("model_id")
                baseline_model.pop("config")
                variant_model.pop("model_id")
                variant_model.pop("config")
                self.assertEqual(variant_model, baseline_model)
                variant["model"] = baseline["model"]
                self.assertEqual(variant, baseline)

    def test_unknown_yaml_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text(
                "\n".join(
                    (
                        'schema_version: "1.0"',
                        "model_route: native-smoke",
                        "run_profile: smoke",
                        "stage: pretrain",
                        "model:",
                        "  provider: native",
                        "  model_id: smoke-10m",
                        "unknown_field: true",
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "unknown run config fields"):
                load_run_config(path)


if __name__ == "__main__":
    unittest.main()
