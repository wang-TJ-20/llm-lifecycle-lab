from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from llm_lifecycle_lab.cli import main


class CliTests(unittest.TestCase):
    def test_model_inspect_reports_vocab_parameter_budget(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "model",
                    "inspect",
                    "--config",
                    str(project_root / "configs/models/smoke-10m.yaml"),
                    "--compare-vocab-size",
                    "8192",
                    "--compare-vocab-size",
                    "16384",
                    "--json",
                ]
            )

        value = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertGreater(value["token_parameter_share"], 0.5)
        self.assertEqual(
            [item["vocab_size"] for item in value["vocab_budget"]],
            [8192, 16384],
        )

    def test_config_validate_and_run_create(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            runs_path = root / "runs"
            config_path.write_text(
                "\n".join(
                    (
                        'schema_version: "1.0"',
                        "model_route: native-smoke",
                        "run_profile: smoke",
                        "stage: pretrain",
                        "seed: 42",
                        f"output_dir: {runs_path}",
                        "model:",
                        "  provider: native",
                        "  model_id: smoke-10m",
                        "training:",
                        "  max_steps: 1",
                        "  sequence_length: 8",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                validate_code = main(["config", "validate", str(config_path)])
                create_code = main(
                    [
                        "run",
                        "create",
                        "--config",
                        str(config_path),
                        "--run-id",
                        "integration-run",
                    ]
                )

            self.assertEqual(validate_code, 0)
            self.assertEqual(create_code, 0)
            self.assertTrue((runs_path / "integration-run/run_manifest.json").is_file())

    def test_data_validate_returns_nonzero_for_invalid_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dpo.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "d1",
                        "prompt": "question",
                        "chosen": "same",
                        "rejected": "same",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "data",
                        "validate",
                        "--input",
                        str(path),
                        "--kind",
                        "dpo",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("identical_pair", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
