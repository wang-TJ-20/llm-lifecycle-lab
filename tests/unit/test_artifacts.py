from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_lifecycle_lab.artifacts import ArtifactStore
from llm_lifecycle_lab.contracts import (
    ModelRoute,
    RunConfig,
    RunProfile,
    Stage,
)
from llm_lifecycle_lab.exceptions import ArtifactError


def _config() -> RunConfig:
    return RunConfig(
        model_route=ModelRoute.NATIVE_SMOKE,
        run_profile=RunProfile.SMOKE,
        stage=Stage.PRETRAIN,
        model={"provider": "native", "model_id": "smoke-10m"},
    )


class ArtifactStoreTests(unittest.TestCase):
    def test_create_run_writes_resolved_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = ArtifactStore(Path(temporary) / "runs").create_run(
                _config(),
                run_id="test-run",
            )

            self.assertTrue((artifacts.path / "resolved_config.yaml").is_file())
            self.assertTrue((artifacts.path / "metrics.jsonl").is_file())
            manifest = json.loads(
                (artifacts.path / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["run_id"], "test-run")
            self.assertEqual(manifest["model_route"], "native-smoke")

    def test_create_run_refuses_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore(Path(temporary) / "runs")
            store.create_run(_config(), run_id="same-run")

            with self.assertRaisesRegex(ArtifactError, "already exists"):
                store.create_run(_config(), run_id="same-run")

    def test_artifact_path_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = ArtifactStore(Path(temporary) / "runs").create_run(
                _config(),
                run_id="test-run",
            )

            with self.assertRaisesRegex(ArtifactError, "inside"):
                artifacts.write_text("../outside.txt", "unsafe")

    def test_metrics_reject_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = ArtifactStore(Path(temporary) / "runs").create_run(
                _config(),
                run_id="test-run",
            )

            with self.assertRaisesRegex(
                ArtifactError,
                "not JSON serializable",
            ):
                artifacts.append_metric({"loss": float("nan")})


if __name__ == "__main__":
    unittest.main()
