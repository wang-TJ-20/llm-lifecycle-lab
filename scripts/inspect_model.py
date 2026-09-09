"""Inspect Native model dimensions and vocabulary parameter budgets."""

from llm_lifecycle_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(command=("model", "inspect"), prog="python scripts/inspect_model.py")
    )
