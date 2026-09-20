"""Summarize exact-match failure modes from capability report JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/analyze_capability_outputs.py",
        description=__doc__,
    )
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _answer_text(rule: dict[str, Any]) -> str:
    answer = rule["answer"]
    if isinstance(answer, str):
        return answer
    return json.dumps(
        answer,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def summarize(report: dict[str, Any], path: Path) -> dict[str, Any]:
    exact_total = 0
    exact_success = 0
    answer_contained = 0
    extra_text = 0
    output_characters = 0
    json_total = 0
    json_parseable = 0
    by_kind: dict[str, dict[str, int]] = {}

    for sample in report["samples"]:
        if sample["kind"] in {"corpus", "continuation", "preference", "memory"}:
            continue
        turns = sample["turns"]
        for turn in turns:
            output = str(turn["output"]["text"]).strip()
            answer = _answer_text(turn["rule"]).strip()
            success = int(float(turn["score"]) == 1.0)
            contained = int(bool(answer) and answer in output)
            row = by_kind.setdefault(
                sample["kind"],
                {"total": 0, "exact_success": 0, "answer_contained": 0},
            )
            row["total"] += 1
            row["exact_success"] += success
            row["answer_contained"] += contained
            exact_total += 1
            exact_success += success
            answer_contained += contained
            extra_text += int(contained and not success)
            output_characters += len(output)
            if turn["rule"]["type"] == "json":
                json_total += 1
                try:
                    json.loads(output)
                except (json.JSONDecodeError, TypeError):
                    pass
                else:
                    json_parseable += 1

    return {
        "report": str(path),
        "run_id": report["model"]["run_id"],
        "checkpoint_id": report["model"]["checkpoint_id"],
        "protocol_sha256": report["protocol_sha256"],
        "exact_total": exact_total,
        "exact_success": exact_success,
        "answer_contained": answer_contained,
        "extra_text": extra_text,
        "mean_output_characters": (
            output_characters / exact_total if exact_total else 0.0
        ),
        "json_total": json_total,
        "json_parseable": json_parseable,
        "by_kind": dict(sorted(by_kind.items())),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        print(f"error: output exists: {args.output}", file=sys.stderr)
        return 2
    try:
        reports = [
            (path, json.loads(path.read_text(encoding="utf-8")))
            for path in args.report
        ]
        protocol_hashes = {report["protocol_sha256"] for _, report in reports}
        if len(protocol_hashes) != 1:
            raise ValueError("reports use different capability protocols")
        result = {
            "protocol_sha256": next(iter(protocol_hashes)),
            "checkpoints": [
                summarize(report, path) for path, report in reports
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
