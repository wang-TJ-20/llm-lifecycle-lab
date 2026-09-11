from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_lifecycle_lab.contracts import (
    ModelRoute,
    RunConfig,
    RunProfile,
    Stage,
)
from llm_lifecycle_lab.data import prepare_dataset
from llm_lifecycle_lab.doctor import profile_for_config, run_doctor
from llm_lifecycle_lab.doctor.result import CheckStatus


class DoctorTests(unittest.TestCase):
    def test_profile_is_derived_from_model_route_and_run_profile(self) -> None:
        qwen = RunConfig(
            model_route=ModelRoute.QWEN3_TRANSFER,
            run_profile=RunProfile.LEARN,
            stage=Stage.SFT,
            model={
                "provider": "huggingface",
                "model_id": "Qwen/Qwen3-0.6B-Base",
                "revision": "a" * 40,
                "tokenizer_revision": "a" * 40,
                "transformers_version": "4.51.3",
            },
        )
        smoke = RunConfig(
            model_route=ModelRoute.NATIVE,
            run_profile=RunProfile.SMOKE,
            stage=Stage.PRETRAIN,
            model={"provider": "native", "model_id": "smoke-10m"},
        )
        learn = RunConfig(
            model_route=ModelRoute.NATIVE,
            run_profile=RunProfile.LEARN,
            stage=Stage.PRETRAIN,
            model={"provider": "native", "model_id": "tiny-60m"},
        )

        self.assertEqual(profile_for_config(qwen), "qwen3-0.6b-base")
        self.assertEqual(profile_for_config(smoke), "smoke-10m")
        self.assertEqual(profile_for_config(learn), "tiny-60m")

    def test_explicit_conflicting_profile_fails(self) -> None:
        config = RunConfig(
            model_route=ModelRoute.NATIVE,
            run_profile=RunProfile.SMOKE,
            stage=Stage.PRETRAIN,
            model={"provider": "native", "model_id": "smoke-10m"},
        )

        report = run_doctor(
            profile="base",
            config=config,
        )
        result = next(
            check for check in report.checks if check.name == "profile-config-match"
        )

        self.assertIs(result.status, CheckStatus.FAIL)

    def test_prepared_data_manifest_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.jsonl"
            source.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": f"p{index}",
                            "text": f"sample {index}",
                            "source": "fixture",
                        }
                    )
                    + "\n"
                    for index in range(40)
                ),
                encoding="utf-8",
            )
            prepared = root / "prepared"
            prepare_dataset(
                source,
                prepared,
                dataset_id="fixture-v1",
                record_kind="pretrain",
                license_name="test-only",
            )
            config = RunConfig(
                model_route=ModelRoute.NATIVE,
                run_profile=RunProfile.SMOKE,
                stage=Stage.PRETRAIN,
                model={"provider": "native", "model_id": "smoke-10m"},
                output_dir=str(root / "runs"),
                data={"manifest": str(prepared / "data_manifest.json")},
            )

            report = run_doctor(config=config, workdir=root)
            result = next(
                check for check in report.checks if check.name == "data-manifest"
            )

            self.assertIs(result.status, CheckStatus.PASS)


if __name__ == "__main__":
    unittest.main()
