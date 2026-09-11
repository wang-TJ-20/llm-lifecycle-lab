from __future__ import annotations

import json
import platform
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from llm_lifecycle_lab.config import load_run_config
from llm_lifecycle_lab.data import prepare_dataset
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.packing import materialize_packed_pretraining_dataset
from llm_lifecycle_lab.exceptions import ConfigError
from llm_lifecycle_lab.provenance import source_tree_sha256
from llm_lifecycle_lab.reference import (
    reference_execution_sha256,
    verify_reference_inputs,
)
from llm_lifecycle_lab.tokenizer import train_native_tokenizer
from llm_lifecycle_lab.training.engine import EngineConfig


class ReferenceInputTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source_dir = self.root / "src/llm_lifecycle_lab"
        source_dir.mkdir(parents=True)
        (source_dir / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
        source = self.root / "source.jsonl"
        records = [
            {
                "id": f"doc-{index}",
                "source": "test-only",
                "text": (
                    f"A short training example with number {index}."
                    if index % 2 == 0
                    else (
                        f"\u4e2d\u6587\u8bad\u7ec3\u793a\u4f8b {index}"
                        " \u7684\u6587\u672c\u5185\u5bb9\u3002"
                    )
                ),
                "language": "en" if index % 2 == 0 else "zh",
            }
            for index in range(60)
        ]
        source.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        prepare_dataset(
            source,
            self.root / "prepared",
            dataset_id="baseline-test",
            record_kind="pretrain",
            license_name="test-only",
            seed=3,
        )
        manifest = self.root / "prepared/data_manifest.json"
        tokenizer = train_native_tokenizer(
            manifest,
            self.root / "tokenizer",
            tokenizer_id="baseline-test",
            vocab_size=320,
            min_frequency=1,
        )
        packed = materialize_packed_pretraining_dataset(
            manifest, self.root / "tokenizer", self.root / "packed", sequence_length=16
        )
        self.model = {
            "model_id": "baseline-test",
            "vocab_size": tokenizer.vocab_size,
            "num_hidden_layers": 1,
            "hidden_size": 24,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "intermediate_size": 48,
            "max_sequence_length": 16,
        }
        self.write_yaml("model.yaml", self.model)
        self.pipeline = {
            "schema_version": "1.0",
            "model_route": "native",
            "run_profile": "smoke",
            "stage": "pretrain",
            "seed": 11,
            "output_dir": "runs",
            "model": {
                "provider": "native",
                "model_id": "baseline-test",
                "config": "model.yaml",
                "tokenizer": "tokenizer",
            },
            "data": {
                "manifest": "prepared/data_manifest.json",
                "packed_manifest": "packed/packed_manifest.json",
            },
            "training": {
                "device": "cpu",
                "dtype": "float32",
                "sequence_length": 16,
                "max_steps": 2,
                "learning_rate": 0.001,
                "eval_batches": 2,
                "checkpoint_interval": 1,
                "eval_interval": 1,
            },
        }
        self.write_yaml("pipeline.yaml", self.pipeline)
        config = load_run_config(self.root / "pipeline.yaml")
        _, budget = EngineConfig.from_dict(config.training).resolve_budget(
            examples_per_epoch=packed.splits[0].examples,
            supervised_tokens_per_epoch=packed.splits[0].supervised_tokens,
        )
        self.spec = {
            "schema_version": "1.0",
            "reference_id": "test-only-reference",
            "pipeline_config": "pipeline.yaml",
            "freeze": {
                "execution_sha256": reference_execution_sha256(
                    config, workdir=self.root
                ),
                "source_sha256": source_tree_sha256(self.root),
                "packed_manifest_sha256": sha256_file(
                    self.root / "packed/packed_manifest.json"
                ),
                "python_major_minor": ".".join(
                    platform.python_version().split(".")[:2]
                ),
                "packages": {"tokenizers": "0.21.4"},
            },
            "requirements": {
                "expected_data_manifest_sha256": sha256_file(manifest),
                "expected_tokenizer_sha256": tokenizer.content_sha256,
                "expected_model_config_sha256": sha256_file(self.root / "model.yaml"),
                "expected_packed_sequence_length": 16,
                "expected_packed_array_sha256": {
                    array.path: array.sha256
                    for split in packed.splits
                    for array in (
                        split.token_ids,
                        split.language_ids,
                        split.byte_weights,
                    )
                },
                "expected_budget_mode": "max_steps",
                "expected_global_step": 2,
                "expected_target_train_tokens": budget.target_train_tokens,
                "minimum_target_token_coverage": 1.0,
                "maximum_target_token_coverage": 1.1,
                "platform_system": platform.system(),
                "device_type": "cpu",
                "minimum_device_memory_gib": 0,
                "require_clean_git": False,
                "require_eval_improvement": False,
                "evaluation_suites": ["pretrain-dev"],
                "required_training_metric_keys": ["train_loss"],
                "required_evaluation_metric_keys": ["eval_loss"],
            },
        }
        self.write_yaml("reference.yaml", self.spec)

    def write_yaml(self, name: str, value: dict) -> None:
        (self.root / name).write_text(yaml.safe_dump(value), encoding="utf-8")

    def verify(self, *, runtime: bool = False):
        return verify_reference_inputs(
            self.root / "reference.yaml", workdir=self.root, check_runtime=runtime
        )

    def test_inputs_only_verifies_artifacts_without_creating_a_run(self) -> None:
        report = self.verify()
        self.assertEqual(report.exit_code, 0, report.to_json())
        self.assertEqual(report.scope, "inputs-only")
        self.assertEqual(
            {check.name for check in report.checks},
            {"execution-lock", "input-artifacts", "input-budget", "source-lock"},
        )
        self.assertFalse((self.root / "runs").exists())

    def test_preflight_includes_runtime_and_input_success_is_not_readiness(
        self,
    ) -> None:
        report = self.verify(runtime=True)
        self.assertEqual(report.exit_code, 0, report.to_json())
        self.spec["freeze"]["python_major_minor"] = "0.0"
        self.write_yaml("reference.yaml", self.spec)
        self.assertEqual(self.verify().exit_code, 0)
        self.assertEqual(self.verify(runtime=True).exit_code, 1)

    def test_pipeline_drift_is_rejected_before_loading_data(self) -> None:
        self.pipeline["training"]["learning_rate"] = 0.002
        self.write_yaml("pipeline.yaml", self.pipeline)
        report = self.verify()
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.checks[0].name, "execution-lock")
        self.assertEqual(report.scope, "inputs-only")

    def test_effective_model_defaults_are_frozen(self) -> None:
        original = self.spec["freeze"]["execution_sha256"]
        self.model["initializer_range"] = 0.03
        self.write_yaml("model.yaml", self.model)
        config = load_run_config(self.root / "pipeline.yaml")
        self.assertNotEqual(
            reference_execution_sha256(config, workdir=self.root), original
        )
        self.assertEqual(self.verify().exit_code, 1)

    def test_selected_training_pipeline_must_match(self) -> None:
        self.pipeline["seed"] = 12
        self.write_yaml("other.yaml", self.pipeline)
        report = verify_reference_inputs(
            self.root / "reference.yaml",
            workdir=self.root,
            check_runtime=False,
            config=load_run_config(self.root / "other.yaml"),
        )
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.checks[-1].name, "selected-pipeline")

    def test_source_and_array_tampering_fail(self) -> None:
        (self.root / "src/llm_lifecycle_lab/example.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
        report = self.verify()
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.checks[-1].name, "source-lock")
        (self.root / "packed/train.tokens.i32").write_bytes(b"bad")
        report = self.verify()
        failures = {
            check.name for check in report.checks if check.status.value == "fail"
        }
        self.assertIn("input-artifacts", failures)

    def test_relocating_the_immutable_artifact_bundle_preserves_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "copy"
            shutil.copytree(self.root, target)
            report = verify_reference_inputs(
                target / "reference.yaml", workdir=target, check_runtime=False
            )
            self.assertEqual(report.exit_code, 0, report.to_json())

    def test_missing_or_misspelled_freeze_fields_fail_fast(self) -> None:
        self.spec["freeze"]["execution_sha"] = self.spec["freeze"].pop(
            "execution_sha256"
        )
        self.write_yaml("reference.yaml", self.spec)
        with self.assertRaisesRegex(ConfigError, "freeze"):
            self.verify()

    def test_budget_drift_is_rejected(self) -> None:
        self.spec["requirements"]["expected_global_step"] = 3
        self.write_yaml("reference.yaml", self.spec)
        report = self.verify()
        self.assertEqual(report.exit_code, 1)
        self.assertIn(
            "input-budget",
            [check.name for check in report.checks if check.status.value == "fail"],
        )

    def test_frozen_runtime_versions_and_source_must_match(self) -> None:
        self.spec["freeze"]["packages"]["torch"] = "2.14.0"
        self.write_yaml("reference.yaml", self.spec)
        runtime = {
            "python": {"version": platform.python_version()},
            "packages": {"tokenizers": "0.21.4", "torch": "2.14.0+cu130"},
            "platform": {"system": platform.system()},
            "accelerator": {"type": "cpu"},
            "code": {"source_sha256": self.spec["freeze"]["source_sha256"]},
        }
        with patch(
            "llm_lifecycle_lab.reference.capture_runtime_provenance",
            return_value=runtime,
        ) as capture:
            self.assertEqual(self.verify(runtime=True).exit_code, 0)
            for section, field, wrong_value in (
                ("python", "version", "0.0.1"),
                ("packages", "torch", "2.13.0+cu130"),
                ("packages", "tokenizers", None),
                ("code", "source_sha256", "0" * 64),
            ):
                with self.subTest(section=section, field=field):
                    original = runtime[section][field]
                    runtime[section][field] = wrong_value
                    report = self.verify(runtime=True)
                    self.assertEqual(
                        [c.name for c in report.checks if c.status.value == "fail"],
                        ["frozen-runtime"],
                    )
                    runtime[section][field] = original
            self.assertEqual(capture.call_count, 5)

    def test_cuda_checks_selected_device_and_bf16_support(self) -> None:
        self.pipeline["training"].update(device="cuda", dtype="bfloat16")
        self.write_yaml("pipeline.yaml", self.pipeline)
        self.spec["freeze"]["execution_sha256"] = reference_execution_sha256(
            load_run_config(self.root / "pipeline.yaml"), workdir=self.root
        )
        self.spec["requirements"].update(
            platform_system="Linux",
            device_type="cuda",
            minimum_device_memory_gib=22,
            require_clean_git=True,
        )
        self.write_yaml("reference.yaml", self.spec)
        runtime = {
            "python": {"version": platform.python_version()},
            "packages": {"tokenizers": "0.21.4"},
            "platform": {"system": "Linux"},
            "accelerator": {
                "type": "cuda",
                "selected_device": "cuda",
                "devices": [
                    {"total_memory_bytes": 24 * 1024**3},
                    {"total_memory_bytes": 80 * 1024**3},
                ],
            },
            "code": {
                "source_sha256": self.spec["freeze"]["source_sha256"],
                "commit": "a" * 40,
                "dirty": False,
            },
        }
        with (
            patch(
                "llm_lifecycle_lab.reference.capture_runtime_provenance",
                return_value=runtime,
            ) as capture,
            patch("torch.cuda.is_available", return_value=True) as cuda,
            patch("torch.cuda.is_bf16_supported", return_value=True) as bf16,
        ):
            self.assertEqual(self.verify(runtime=True).exit_code, 0)
            runtime["accelerator"]["devices"][0]["total_memory_bytes"] = 8 * 1024**3
            report = self.verify(runtime=True)
            self.assertEqual(
                [c.name for c in report.checks if c.status.value == "fail"],
                ["runtime-provenance"],
            )
            runtime["accelerator"]["devices"][0]["total_memory_bytes"] = 24 * 1024**3
            runtime["code"]["dirty"] = True
            self.assertEqual(self.verify(runtime=True).exit_code, 1)
            runtime["code"]["dirty"] = False
            bf16.return_value = False
            report = self.verify(runtime=True)
            self.assertEqual(
                [c.name for c in report.checks if c.status.value == "fail"],
                ["cuda-compute"],
            )
            self.assertEqual(capture.call_count, 4)
            self.assertEqual(cuda.call_count, 4)
            self.assertEqual(bf16.call_count, 4)

    def test_evaluation_sample_requirement_is_strict(self) -> None:
        for value in (0, -1, True, 2.5, "64"):
            with self.subTest(value=value):
                self.spec["requirements"]["expected_evaluation_samples"] = value
                self.write_yaml("reference.yaml", self.spec)
                with self.assertRaisesRegex(ConfigError, "expected_evaluation_samples"):
                    self.verify()
