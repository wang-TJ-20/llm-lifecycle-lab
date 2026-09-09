"""Train or resume Native Pretrain using the shared engine."""

from llm_lifecycle_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(command=("train", "pretrain"), prog="python scripts/train_pretrain.py")
    )
