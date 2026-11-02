"""Evaluate full A10 SFT dev, or authorize the one-time test after dev acceptance."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from train_a10_sft import PREFIX, check_clean_git, check_hardware

from llm_lifecycle_lab.config import config_sha256, load_run_config
from llm_lifecycle_lab.contracts import utc_now
from llm_lifecycle_lab.exceptions import ContractError, LLMLabError
from llm_lifecycle_lab.sft_run_gate import (
    _execution_lock_check,
    _load_spec,
    verify_sft_run,
)
from llm_lifecycle_lab.training.sft import evaluate_native_sft


def claim_sealed_test(run: Path, config_sha: str) -> None:
    if any((run / "evaluations").glob("sft-test-*.json")):
        raise ContractError("sealed test has already been evaluated")
    with (run / "sealed-test-started.json").open("x", encoding="utf-8") as handle:
        json.dump({"config_sha256": config_sha, "started_at": utc_now()}, handle)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sealed-test", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    run = args.run if args.run.is_absolute() else root / args.run
    try:
        check_clean_git(root)
        check_hardware()
        config = load_run_config(root / f"configs/pipelines/{PREFIX}.yaml")
        spec_path = root / f"configs/gates/{PREFIX}-run.yaml"
        spec = _load_spec(spec_path, root)
        if _execution_lock_check(spec, root).status.value != "pass":
            raise ContractError("A10 SFT execution lock failed")
        if args.sealed_test:
            report = verify_sft_run(spec_path, run, workdir=root)
            if report.has_failures:
                raise ContractError(
                    "pass completed-run and full-dev acceptance before test"
                )
            claim_sealed_test(run, config_sha256(config))
        result = evaluate_native_sft(
            config,
            checkpoint=run / "checkpoints" / f"step-{spec.training.global_step:08d}",
            split="test" if args.sealed_test else "dev",
            workdir=root,
        )
        print(json.dumps(result, indent=2))
        return 0
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
