"""下载并验证公开模型，然后运行固定续写与双语 Smoke 评测。

Download and verify a published model, then run fixed continuations and a
bilingual smoke evaluation.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.exceptions import ArtifactError, LLMLabError
from llm_lifecycle_lab.release import (
    evaluate_release_smoke,
    generate_release_samples,
    verify_release_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_published_model.py",
        description=(
            "Download a Native release from ModelScope, verify it, generate "
            "fixed continuations, and run a bilingual smoke evaluation."
        ),
    )
    parser.add_argument(
        "--model-id",
        default="llmlifecyclelab/native-60m-base-v1",
    )
    parser.add_argument("--revision", default="master")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("build/downloads/native-60m-base-v1"),
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="use --local-dir without contacting ModelScope",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _download_model(
    model_id: str,
    *,
    revision: str,
    local_dir: Path,
) -> Path:
    try:
        module = importlib.import_module("modelscope")
        snapshot_download = module.snapshot_download
    except (ImportError, AttributeError) as exc:
        raise ArtifactError(
            "ModelScope support is not installed; run "
            "`python -m pip install -r requirements-modelscope.txt`"
        ) from exc

    if local_dir.exists() and any(local_dir.iterdir()):
        raise ArtifactError(
            "download directory is not empty; use a new --local-dir for an "
            f"independent verification: {local_dir}"
        )
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = snapshot_download(
            model_id,
            revision=revision,
            local_dir=str(local_dir),
        )
    except Exception as exc:
        raise ArtifactError(
            f"cannot download ModelScope model {model_id}: {exc}"
        ) from exc
    return Path(downloaded)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        release = (
            args.local_dir
            if args.use_local
            else _download_model(
                args.model_id,
                revision=args.revision,
                local_dir=args.local_dir,
            )
        )
        integrity = verify_release_files(release)
        generation = generate_release_samples(
            release,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            verify_hashes=False,
        )
        evaluation = evaluate_release_smoke(
            release,
            device=args.device,
            verify_hashes=False,
        )
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = {
        "model_id": args.model_id,
        "release": str(release),
        "integrity": integrity,
        "generation": generation,
        "evaluation": evaluation,
    }
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print(
        f"verified={integrity['files_verified']} "
        f"model={integrity['repository_id']} "
        f"device={evaluation['device']}"
    )
    print(
        f"smoke: loss={evaluation['eval_loss']:.6f} "
        f"perplexity={evaluation['eval_perplexity']:.4f} "
        f"bits_per_byte={evaluation['eval_bits_per_byte']:.6f}"
    )
    for item in generation["results"]:
        print(f"prompt: {item['prompt']}")
        print(f"  continuation: {item['continuation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
