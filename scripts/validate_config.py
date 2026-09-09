"""Validate a pipeline configuration without starting a run."""

from llm_lifecycle_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(command=("config", "validate"), prog="python scripts/validate_config.py")
    )
