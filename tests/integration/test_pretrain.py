from __future__ import annotations

import json
import math
import platform
import tempfile
import unittest
from pathlib import Path

import yaml

from llm_lifecycle_lab.contracts import (
    ModelRoute,
    RunConfig,
    RunProfile,
    Stage,
)
from llm_lifecycle_lab.data import prepare_dataset
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.packing import (
    materialize_packed_pretraining_dataset,
)
from llm_lifecycle_lab.doctor import run_doctor
from llm_lifecycle_lab.doctor.result import CheckStatus
from llm_lifecycle_lab.reference import verify_reference_run
from llm_lifecycle_lab.tokenizer import train_native_tokenizer
from llm_lifecycle_lab.training.pretrain import (
    evaluate_native_pretraining,
    run_native_pretraining,
)


class PretrainingIntegrationTests(unittest.TestCase):
    def test_two_step_pretraining_writes_metrics_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_manifest = _prepare_data(root)
            tokenizer_dir = root / "tokenizer"
            tokenizer_manifest = train_native_tokenizer(
                data_manifest,
                tokenizer_dir,
                tokenizer_id="integration-bpe",
                vocab_size=320,
                min_frequency=1,
            )
            packed_dir = root / "packed"
            packed_manifest = materialize_packed_pretraining_dataset(
                data_manifest,
                tokenizer_dir,
                packed_dir,
                sequence_length=16,
            )
            model_config_path = root / "model.yaml"
            model_config_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "1.0",
                        "model_route": "native-smoke",
                        "provider": "native",
                        "model_id": "integration-micro",
                        "architecture": "dense-decoder",
                        "vocab_size": tokenizer_manifest.vocab_size,
                        "num_hidden_layers": 1,
                        "hidden_size": 24,
                        "num_attention_heads": 3,
                        "num_key_value_heads": 1,
                        "intermediate_size": 48,
                        "max_sequence_length": 16,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            config = RunConfig(
                model_route=ModelRoute.NATIVE_SMOKE,
                run_profile=RunProfile.SMOKE,
                stage=Stage.PRETRAIN,
                seed=11,
                output_dir=str(root / "runs"),
                model={
                    "provider": "native",
                    "model_id": "integration-micro",
                    "config": str(model_config_path),
                    "tokenizer": str(tokenizer_dir),
                },
                data={
                    "manifest": str(data_manifest),
                    "packed_manifest": str(packed_dir / "packed_manifest.json"),
                },
                training={
                    "device": "cpu",
                    "dtype": "float32",
                    "sequence_length": 16,
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "max_steps": 2,
                    "learning_rate": 0.001,
                    "warmup_steps": 0,
                    "checkpoint_interval": 1,
                    "eval_interval": 1,
                    "eval_batches": 1,
                    "log_interval": 1,
                },
            )
            pipeline_path = root / "pipeline.yaml"
            pipeline_path.write_text(
                yaml.safe_dump(config.to_dict(), sort_keys=True),
                encoding="utf-8",
            )

            doctor_report = run_doctor(config=config, workdir=root)
            real_batch = next(
                check for check in doctor_report.checks if check.name == "real-batch"
            )
            packed_data = next(
                check for check in doctor_report.checks if check.name == "packed-data"
            )
            self.assertIs(real_batch.status, CheckStatus.PASS)
            self.assertIs(packed_data.status, CheckStatus.PASS)

            run = run_native_pretraining(
                config,
                run_id="integration-pretrain",
                workdir=root,
            )

            self.assertEqual(run.result.global_step, 2)
            self.assertGreater(run.result.tokens_seen, 0)
            self.assertGreater(run.result.target_train_tokens, 0)
            self.assertGreater(run.result.epochs_seen, 0)
            self.assertGreater(run.result.final_loss, 0)
            self.assertTrue(
                (
                    run.artifacts.path / "checkpoints/step-00000002/model/model.pt"
                ).is_file()
            )
            self.assertTrue((run.artifacts.path / "tokenizer/tokenizer.json").is_file())
            self.assertTrue((run.artifacts.path / "model_config.json").is_file())
            self.assertTrue(
                (run.artifacts.path / "packed_data_snapshot.json").is_file()
            )
            self.assertTrue((run.artifacts.path / "runtime_environment.json").is_file())
            metrics = (
                (run.artifacts.path / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(len(metrics), 3)
            baseline = json.loads(metrics[0])
            self.assertEqual(baseline["event"], "baseline")
            self.assertGreater(baseline["eval_bits_per_byte"], 0)
            self.assertGreater(baseline["eval_en_loss"], 0)
            self.assertGreater(baseline["eval_zh_loss"], 0)
            final_metrics = json.loads(metrics[-1])
            self.assertGreater(final_metrics["train_bits_per_byte"], 0)
            self.assertGreater(final_metrics["epochs_seen"], 0)
            budget = json.loads(
                (run.artifacts.path / "training_budget.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(budget["mode"], "max_steps")
            self.assertEqual(budget["max_steps"], 2)
            run_manifest = json.loads(
                (run.artifacts.path / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_manifest["status"], "completed")

            evaluation = evaluate_native_pretraining(
                config,
                checkpoint=run.result.final_checkpoint,
                split="dev",
                workdir=root,
            )
            self.assertEqual(evaluation["checkpoint_step"], 2)
            self.assertGreater(evaluation["eval_loss"], 0)
            self.assertGreater(evaluation["eval_bits_per_byte"], 0)
            language_weighted_loss = (
                evaluation["eval_en_loss"] * evaluation["eval_en_tokens"]
                + evaluation["eval_zh_loss"] * evaluation["eval_zh_tokens"]
            ) / (evaluation["eval_en_tokens"] + evaluation["eval_zh_tokens"])
            self.assertEqual(
                evaluation["eval_tokens"],
                evaluation["eval_en_tokens"] + evaluation["eval_zh_tokens"],
            )
            # Float32 reductions can differ across CPU backends.
            self.assertTrue(
                math.isclose(
                    evaluation["eval_loss"],
                    language_weighted_loss,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ),
                msg=(
                    f"aggregate={evaluation['eval_loss']}, "
                    f"buckets={language_weighted_loss}"
                ),
            )
            self.assertTrue(Path(evaluation["report_path"]).is_file())

            reference_spec = root / "reference.yaml"
            reference_spec.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "1.0",
                        "reference_id": "integration-reference",
                        "pipeline_config": str(pipeline_path),
                        "requirements": {
                            "expected_data_manifest_sha256": sha256_file(data_manifest),
                            "expected_tokenizer_sha256": (
                                tokenizer_manifest.content_sha256
                            ),
                            "expected_model_config_sha256": sha256_file(
                                model_config_path
                            ),
                            "expected_packed_sequence_length": 16,
                            "expected_packed_array_sha256": {
                                array.path: array.sha256
                                for split in packed_manifest.splits
                                for array in (
                                    split.token_ids,
                                    split.language_ids,
                                    split.byte_weights,
                                )
                            },
                            "expected_budget_mode": "max_steps",
                            "expected_global_step": 2,
                            "expected_target_train_tokens": (
                                run.result.target_train_tokens
                            ),
                            "minimum_target_token_coverage": 1.0,
                            "maximum_target_token_coverage": 1.1,
                            "platform_system": platform.system(),
                            "device_type": "cpu",
                            "minimum_device_memory_gib": 0,
                            "require_clean_git": False,
                            "require_eval_improvement": False,
                            "evaluation_suites": ["pretrain-dev"],
                            "required_training_metric_keys": [
                                "train_loss",
                                "eval_loss",
                                "eval_en_loss",
                                "eval_en_tokens",
                                "eval_zh_loss",
                                "eval_zh_tokens",
                            ],
                            "required_evaluation_metric_keys": [
                                "eval_loss",
                                "eval_tokens",
                                "eval_en_loss",
                                "eval_en_tokens",
                                "eval_zh_loss",
                                "eval_zh_tokens",
                            ],
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            reference_report = verify_reference_run(
                reference_spec,
                run.artifacts.path,
                workdir=root,
            )
            self.assertFalse(reference_report.has_failures)


def _prepare_data(root: Path) -> Path:
    source = root / "source.jsonl"
    records = [
        {
            "id": f"doc-{index}",
            "text": (
                f"Training sample {index}. "
                f"The result of {index} plus one is {index + 1}."
            ),
            "source": "integration-test",
            "language": "en" if index % 2 == 0 else "zh",
        }
        for index in range(60)
    ]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    prepared = root / "prepared"
    prepare_dataset(
        source,
        prepared,
        dataset_id="integration-pretrain",
        record_kind="pretrain",
        license_name="test-only",
        seed=3,
    )
    return prepared / "data_manifest.json"


if __name__ == "__main__":
    unittest.main()
