"""构建可发布的 Native 模型目录并生成完整校验信息。

Build a publishable Native model directory with complete integrity metadata.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.release import build_native_model_release


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_model_release.py",
        description="Build a ModelScope-ready Native model release directory.",
    )
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizers/bilingual-60m-v1"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("docs/experiments/results/native-60m-baseline-v1"),
    )
    parser.add_argument(
        "--reference-spec",
        type=Path,
        default=Path("configs/reference/native-60m-baseline-v1.yaml"),
    )
    parser.add_argument("--license", type=Path, default=Path("LICENSE"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/modelscope/native-60m-base-v1"),
    )
    parser.add_argument(
        "--repository-id",
        default="llmlifecyclelab/native-60m-base-v1",
    )
    parser.add_argument("--model-id", default="native-60m-base-v1")
    parser.add_argument("--license-id", default="Apache-2.0")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_native_model_release(
            checkpoint_archive=args.checkpoint_archive,
            tokenizer_dir=args.tokenizer,
            evidence_dir=args.evidence,
            reference_spec=args.reference_spec,
            license_path=args.license,
            output_dir=args.output,
            repository_id=args.repository_id,
            model_id=args.model_id,
            license_id=args.license_id,
            device=args.device,
        )
    except (LLMLabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output": str(args.output),
                "repository_id": manifest["repository_id"],
                "checkpoint_id": manifest["checkpoint_id"],
                "checkpoint_archive_sha256": (manifest["checkpoint_archive"]["sha256"]),
                "files": len(manifest["files"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
