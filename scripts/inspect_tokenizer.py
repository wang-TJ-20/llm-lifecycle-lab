"""Inspect Native tokenizer encoding and decoding."""

from llm_lifecycle_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            command=("tokenizer", "inspect"),
            prog="python scripts/inspect_tokenizer.py",
        )
    )
