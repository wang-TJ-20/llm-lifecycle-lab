"""Fetch, mix, validate, prepare, and pack pretraining data."""

from llm_lifecycle_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(command=("data",), prog="python scripts/data.py"))
