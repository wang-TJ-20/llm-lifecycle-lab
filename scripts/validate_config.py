"""校验一份训练流程配置，不启动训练。

Validate one pipeline configuration without starting a training run.
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

from llm_lifecycle_lab.config import config_sha256, load_run_config
from llm_lifecycle_lab.contracts import ModelRoute, Stage
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.training.engine import EngineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/validate_config.py",
        description="Validate a pipeline YAML without starting a run.",
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_run_config(args.path)
        if config.stage is Stage.PRETRAIN and config.model_route is ModelRoute.NATIVE:
            EngineConfig.from_dict(config.training)
    except (LLMLabError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = {
        "ok": True,
        "path": str(args.path),
        "sha256": config_sha256(config),
        "config": config.to_dict(),
    }
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"PASS config {args.path}")
        print(f"  sha256: {result['sha256']}")
        print(f"  route: {config.model_route.value}")
        print(f"  profile: {config.run_profile.value}")
        print(f"  stage: {config.stage.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
