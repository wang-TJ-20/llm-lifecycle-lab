"""Check the environment and a real batch before training."""

from llm_lifecycle_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(command=("doctor",), prog="python scripts/doctor.py"))
