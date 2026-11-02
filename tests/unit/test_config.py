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

    def test_a10_pipelines_share_model_and_data_contracts(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        pipeline_root = project_root / "configs/pipelines"
        baseline = load_run_config(
            pipeline_root / "native-60m-a10-base-v1.yaml"
        ).to_dict()
        development = (
            "native-60m-a10-base-smoke.yaml",
            "native-60m-a10-base-pilot100.yaml",
        )

        for name in development:
            with self.subTest(name=name):
                config = load_run_config(pipeline_root / name).to_dict()
                self.assertEqual(config["model"], baseline["model"])
                self.assertEqual(config["data"], baseline["data"])
                self.assertEqual(config["seed"], baseline["seed"])
                for key in (
                    "device",
                    "dtype",
                    "sequence_length",
                    "micro_batch_size",
                    "gradient_accumulation_steps",
                ):
                    self.assertEqual(config["training"][key], baseline["training"][key])

    def test_unknown_yaml_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text(
                "\n".join(
                    (
                        'schema_version: "1.0"',
                        "model_route: native",
                        "run_profile: smoke",
                        "stage: pretrain",
                        "model:",
                        "  provider: native",
                        "  model_id: test-micro",
                        "unknown_field: true",
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "unknown run config fields"):
                load_run_config(path)


if __name__ == "__main__":
    unittest.main()
