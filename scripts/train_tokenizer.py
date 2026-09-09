"""Train the versioned Native BPE tokenizer."""

from llm_lifecycle_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(command=("tokenizer", "train"), prog="python scripts/train_tokenizer.py")
    )
