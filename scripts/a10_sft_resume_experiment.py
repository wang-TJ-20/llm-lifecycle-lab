"""User-operated exact-resume experiment on the eight-step A10 SFT smoke."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from _project_path import add_project_src_to_path

add_project_src_to_path()

from a10_resume_experiment import (
    PlannedInterruption,
    interrupt_after_fifth_update,
    state_equal,
)
from train_a10_sft import PREFIX, check_clean_git, check_hardware

from llm_lifecycle_lab.artifacts import RunArtifacts
from llm_lifecycle_lab.config import config_sha256, load_run_config
from llm_lifecycle_lab.evaluation.suite import read_json
from llm_lifecycle_lab.exceptions import ContractError, LLMLabError
from llm_lifecycle_lab.training.sft import run_native_sft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-run", type=Path, default=Path(f"runs/{PREFIX}-smoke-s42")
    )
    parser.add_argument("--run-id", default=f"{PREFIX}-resume-s42")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    baseline = args.baseline_run
    if not baseline.is_absolute():
        baseline = root / baseline
    try:
        if args.run_id != f"{PREFIX}-resume-s42":
            raise ContractError(
                "the frozen formal gate requires the declared resume run ID"
            )
        check_clean_git(root)
        check_hardware()
        config = load_run_config(root / f"configs/pipelines/{PREFIX}-smoke.yaml")
        manifest = read_json(baseline / "run_manifest.json")
        if (
            manifest["status"] != "completed"
            or manifest["config_sha256"] != config_sha256(config)
            or read_json(baseline / "training_result.json")["global_step"] != 8
        ):
            raise ContractError("baseline must be the completed frozen A10 SFT smoke")
        run_path = root / "runs" / args.run_id
        if run_path.exists():
            raise ContractError(
                "resume experiment exists; do not overwrite its evidence"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        try:
            run_native_sft(
                config,
                run_id=args.run_id,
                workdir=root,
                metric_callback=interrupt_after_fifth_update,
            )
        except PlannedInterruption:
            pass
        else:
            raise ContractError("controlled interruption did not occur")
        latest = read_json(run_path / "latest_checkpoint.json")
        if latest["step"] != 4:
            raise ContractError("expected the complete step-four checkpoint")
        resumed = run_native_sft(config, resume_run=args.run_id, workdir=root)
        left = baseline / "checkpoints/step-00000008"
        right = Path(resumed.result.final_checkpoint)
        checks = {
            key: state_equal(
                torch.load(left / file, map_location="cpu", weights_only=True),
                torch.load(right / file, map_location="cpu", weights_only=True),
            )
            for key, file in (
                ("weights_equal", "model/model.pt"),
                ("optimizer_scheduler_rng_equal", "optimizer_state.pt"),
            )
        }
        checks["trainer_and_stream_equal"] = state_equal(
            read_json(left / "trainer_state.json"),
            read_json(right / "trainer_state.json"),
        )
        report = {
            "status": "passed" if all(checks.values()) else "failed",
            "baseline_run": baseline.name,
            "run_id": args.run_id,
            "config_sha256": config_sha256(config),
            "parent_weights_sha256": read_json(run_path / "initialization.json")[
                "parent_weights_sha256"
            ],
            "restored_checkpoint_step": 4,
            "final_step": resumed.result.global_step,
            "scope": "same A10 BF16 and frozen SFT inputs; not cross-device equality",
            **checks,
        }
        RunArtifacts(run_id=args.run_id, path=run_path).write_json(
            "a10-sft-resume-comparison.json", report
        )
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "passed" else 1
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
