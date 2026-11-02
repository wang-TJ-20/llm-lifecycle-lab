"""User-operated A10 SFT launcher; check-only never starts a training run."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from _project_path import add_project_src_to_path

add_project_src_to_path()

from train_sft import print_metric

from llm_lifecycle_lab.config import config_sha256, load_run_config
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.evaluation.suite import read_json
from llm_lifecycle_lab.exceptions import ContractError, LLMLabError
from llm_lifecycle_lab.sft_run_gate import (
    _execution_lock_check,
    _initialization_check,
    _load_spec,
)
from llm_lifecycle_lab.training.sft import _verify_gates, run_native_sft

PREFIX = "native-60m-a10-sft-v1"


def check_hardware() -> str:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ContractError("this SFT route requires exactly one CUDA A10")
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA A10" or not torch.cuda.is_bf16_supported():
        raise ContractError(f"expected NVIDIA A10 with BF16 support, found {name}")
    return name


def check_clean_git(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ContractError(
            "commit SFT code, configs and documentation before training"
        )


def check_development(root: Path, spec) -> None:
    for phase, steps in (("smoke", 8), ("pilot100", 100)):
        run = root / "runs" / f"{PREFIX}-{phase}-s42"
        if not (run / "training_result.json").is_file():
            raise ContractError(
                f"complete and inspect {phase} before formal SFT: {run}"
            )
        config = load_run_config(root / f"configs/pipelines/{PREFIX}-{phase}.yaml")
        manifest = read_json(run / "run_manifest.json")
        result = read_json(run / "training_result.json")
        initialization = read_json(run / "initialization.json")
        latest = read_json(run / "latest_checkpoint.json")
        checkpoint = run / "checkpoints" / f"step-{steps:08d}"
        parent_check = _initialization_check(spec, initialization)
        if (
            manifest["status"] != "completed"
            or manifest["config_sha256"] != config_sha256(config)
            or result["global_step"] != steps
            or latest["step"] != steps
            or parent_check.status.value != "pass"
            or not all(
                math.isfinite(float(result[key])) and float(result[key]) > 0
                for key in ("final_loss", "best_eval_loss", "tokens_seen")
            )
            or sha256_file(checkpoint / "model/model.pt")
            == spec.identity.parent_weights_sha256
        ):
            raise ContractError(f"{phase} evidence is missing, drifted or invalid")
    resume = root / "runs" / f"{PREFIX}-resume-s42" / "a10-sft-resume-comparison.json"
    if not resume.is_file():
        raise ContractError(
            "complete the A10 SFT exact-resume experiment before formal SFT"
        )
    report = read_json(resume)
    if (
        report.get("status") != "passed"
        or report.get("config_sha256")
        != config_sha256(
            load_run_config(root / f"configs/pipelines/{PREFIX}-smoke.yaml")
        )
        or report.get("parent_weights_sha256") != spec.identity.parent_weights_sha256
        or not all(
            report.get(key) is True
            for key in (
                "weights_equal",
                "optimizer_scheduler_rng_equal",
                "trainer_and_stream_equal",
            )
        )
    ):
        raise ContractError(
            "A10 SFT exact-resume evidence differs from the frozen inputs"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("smoke", "pilot100", "formal"), default="smoke"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-id")
    group.add_argument("--resume-run")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    suffix = "" if args.phase == "formal" else f"-{args.phase}"
    try:
        if args.resume_checkpoint and not args.resume_run:
            raise ContractError("--resume-checkpoint requires --resume-run")
        check_clean_git(root)
        hardware = check_hardware()
        config = load_run_config(root / f"configs/pipelines/{PREFIX}{suffix}.yaml")
        spec = _load_spec(root / f"configs/gates/{PREFIX}-run.yaml", root)
        lock = _execution_lock_check(spec, root)
        if lock.status.value != "pass":
            raise ContractError(f"formal SFT execution lock failed: {lock.details}")
        _verify_gates(config, root)
        if args.phase == "formal" and not args.resume_run:
            check_development(root, spec)
        print(f"Preflight passed: {hardware}, phase={args.phase}", flush=True)
        if args.check_only:
            print("Check only: no training started.")
            return 0
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        run = run_native_sft(
            config,
            run_id=args.run_id,
            resume_run=args.resume_run,
            resume_checkpoint=args.resume_checkpoint,
            workdir=root,
            metric_callback=print_metric,
        )
        print(f"Completed {run.artifacts.run_id}: {run.result.global_step} steps")
        return 0
    except (
        LLMLabError,
        OSError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
