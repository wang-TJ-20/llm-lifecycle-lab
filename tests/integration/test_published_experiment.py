from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "docs/experiments/results/native-60m-baseline-v1"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_reference_acceptance_matches_recorded_provenance() -> None:
    acceptance = _json(RESULTS / "acceptance.json")
    runtime = _json(RESULTS / "runtime_environment.json")
    spec = yaml.safe_load(
        (PROJECT_ROOT / "configs/reference/native-60m-baseline-v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert acceptance["decision"] == "accepted-with-provenance-waiver"
    assert acceptance["reference_id"] == acceptance["run_id"]
    assert acceptance["waived_check"] == "runtime-provenance"
    assert acceptance["waiver_scope"] == {
        "git_commit": runtime["code"]["commit"],
        "git_dirty": runtime["code"]["dirty"],
        "source_sha256": runtime["code"]["source_sha256"],
        "status_entries": runtime["code"]["status_entries"],
    }
    assert acceptance["future_runs_require_clean_git"] is True
    assert spec["requirements"]["require_clean_git"] is True


def test_published_metrics_match_budget_and_language_totals() -> None:
    rows = [
        json.loads(line)
        for line in (RESULTS / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = rows[0]
    final = next(row for row in reversed(rows) if row.get("step") == 5649)
    budget = _json(RESULTS / "training_budget.json")
    result = _json(RESULTS / "training_result.json")

    assert baseline["event"] == "baseline"
    assert result["global_step"] == budget["max_steps"] == 5649
    assert result["tokens_seen"] == 46_186_063
    assert all(
        final[key] < baseline[key]
        for key in ("eval_loss", "eval_en_loss", "eval_zh_loss")
    )

    for metrics in (
        baseline,
        final,
        _json(RESULTS / "evaluations/pretrain-dev-step-00005649.json")["metrics"],
        _json(RESULTS / "evaluations/pretrain-test-step-00005649.json")["metrics"],
    ):
        weighted = (
            metrics["eval_en_loss"] * metrics["eval_en_tokens"]
            + metrics["eval_zh_loss"] * metrics["eval_zh_tokens"]
        ) / (metrics["eval_en_tokens"] + metrics["eval_zh_tokens"])
        assert math.isclose(
            weighted,
            metrics["eval_loss"],
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
