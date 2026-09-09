"""Verify a completed run against a versioned reference specification."""

from llm_lifecycle_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            command=("reference", "verify"),
            prog="python scripts/verify_reference.py",
        )
    )
