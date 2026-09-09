"""Evaluate a Native Pretrain checkpoint on the dev or test split."""

from llm_lifecycle_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(command=("eval", "pretrain"), prog="python scripts/eval_pretrain.py")
    )
