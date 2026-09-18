"""Pre-registered fail-fast gates for Native-60M lifecycle stages."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.exceptions import ContractError

_TASK_RETENTION = (
    "instruction.success",
    "format.success",
    "qa.success",
    "multiturn.success",
    "verifiable.reward",
)


def check_base_gate(
    run_dir: str | Path,
    *,
    dev_evaluation: str | Path,
    test_evaluation: str | Path,
    baseline_test_evaluation: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir)
    result = _read_json(run / "training_result.json")
    runtime = _read_json(run / "runtime_environment.json")
    manifest = _read_json(run / "run_manifest.json")
    dev = _evaluation_metrics(_read_json(dev_evaluation))
    test = _evaluation_metrics(_read_json(test_evaluation))
    baseline_test = _evaluation_metrics(_read_json(baseline_test_evaluation))
    checks = [
        _check(
            "run-completed",
            manifest.get("status") == "completed",
            manifest.get("status"),
            "completed",
        ),
        _check(
            "clean-git",
            _mapping(runtime.get("code")).get("dirty") is False,
            _mapping(runtime.get("code")).get("dirty"),
            False,
        ),
        _bounded(
            "target-token-coverage",
            result.get("target_token_coverage"),
            minimum=1.0,
            maximum=1.01,
        ),
        _maximum("dev-loss", dev.get("eval_loss"), 2.9480),
        _maximum("dev-en-loss", dev.get("eval_en_loss"), 2.0119),
        _maximum("dev-zh-loss", dev.get("eval_zh_loss"), 3.8698),
    ]
    for name in ("eval_loss", "eval_en_loss", "eval_zh_loss"):
        baseline = _number(baseline_test.get(name), f"baseline test {name}")
        checks.append(
            _maximum(
                f"test-{name.removeprefix('eval_')}",
                test.get(name),
                baseline * 1.01,
            )
        )
    return _report("base", checks)


def check_sft_gate(
    run_dir: str | Path,
    *,
    capability_report: str | Path,
    parent_capability_report: str | Path,
    data_check: str | Path,
) -> dict[str, Any]:
    baseline, final = _run_metrics(run_dir)
    report = _read_json(capability_report)
    parent = _read_json(parent_capability_report)
    data = _read_json(data_check)
    checks = [
        _relative_decrease(
            "dev-objective-improvement",
            final.get("eval_loss"),
            baseline.get("eval_loss"),
            minimum_fraction=0.05,
        ),
        _minimum_successes(report, "instruction.success", 4),
        _minimum_successes(report, "format.success", 3),
        _minimum_successes(report, "qa.success", 4),
        _minimum_successes(report, "verifiable.reward", 12),
        _maximum_ratio(
            "corpus-bpb-retention",
            _metric_value(report, "corpus.bpb.all"),
            _metric_value(parent, "corpus.bpb.all"),
            1.02,
        ),
        _check(
            "posttraining-data-isolation",
            data.get("ok") is True
            and all(
                int(value) == 0
                for value in _mapping(
                    data.get("stage_source_id_overlap")
                ).values()
            ),
            data.get("stage_source_id_overlap"),
            "all zero",
        ),
    ]
    for task in ("instruction.success", "format.success", "qa.success"):
        for language in ("en", "zh"):
            checks.append(
                _minimum_successes(report, f"{task}.{language}", 1)
            )
    return _report("sft", checks)


def check_dpo_gate(
    run_dir: str | Path,
    *,
    capability_report: str | Path,
    parent_capability_report: str | Path,
) -> dict[str, Any]:
    baseline, final = _run_metrics(run_dir)
    report = _read_json(capability_report)
    parent = _read_json(parent_capability_report)
    accuracy = _number(final.get("eval_pair_accuracy"), "eval_pair_accuracy")
    pairs = int(_number(
        final.get("eval_normalization_count"),
        "eval_normalization_count",
    ))
    correct = int(round(accuracy * pairs))
    if pairs >= 200:
        pair_ok = _wilson_lower(correct, pairs) > 0.5
        pair_expected: Any = "95% Wilson lower bound > 0.5"
        pair_actual: Any = {
            "correct": correct,
            "pairs": pairs,
            "wilson_lower": _wilson_lower(correct, pairs),
        }
    else:
        pair_ok = pairs == 46 and correct >= 30
        pair_expected = "30/46, or >=200 with Wilson lower bound > 0.5"
        pair_actual = {"correct": correct, "pairs": pairs}
    checks = [
        _check(
            "dev-preference-loss-improvement",
            _number(final.get("eval_loss"), "final eval_loss")
            < _number(baseline.get("eval_loss"), "baseline eval_loss"),
            final.get("eval_loss"),
            f"< {baseline.get('eval_loss')}",
        ),
        _check("dev-pair-accuracy", pair_ok, pair_actual, pair_expected),
        _success_delta(
            "independent-preference-gain",
            report,
            parent,
            "preference.accuracy",
            minimum_delta=3,
        ),
        _maximum_ratio(
            "corpus-bpb-retention",
            _metric_value(report, "corpus.bpb.all"),
            _metric_value(parent, "corpus.bpb.all"),
            1.02,
        ),
    ]
    checks.extend(
        _retention_check(report, parent, name) for name in _TASK_RETENTION
    )
    return _report("dpo", checks)


def check_grpo_gate(
    run_dir: str | Path,
    *,
    capability_report: str | Path,
    parent_capability_report: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir)
    baseline, final = _run_metrics(run)
    report = _read_json(capability_report)
    parent = _read_json(parent_capability_report)
    budget = _read_json(run / "grpo_budget_summary.json")
    checks = [
        _check(
            "dev-reward-improvement",
            _number(final.get("eval_reward_mean"), "final eval_reward_mean")
            >= _number(
                baseline.get("eval_reward_mean"),
                "baseline eval_reward_mean",
            )
            + 0.05,
            final.get("eval_reward_mean"),
            f">= {float(baseline['eval_reward_mean']) + 0.05}",
        ),
        _maximum(
            "dev-zero-variance-groups",
            final.get("eval_zero_variance_groups"),
            0.75,
        ),
        _maximum("dev-approx-kl", final.get("eval_approx_kl"), 0.10),
        _maximum_ratio(
            "corpus-bpb-retention",
            _metric_value(report, "corpus.bpb.all"),
            _metric_value(parent, "corpus.bpb.all"),
            1.02,
        ),
        _check(
            "separate-budget-accounting",
            _positive(budget.get("prompt_passes"))
            and _positive(budget.get("rollout_token_coverage")),
            {
                "prompt_passes": budget.get("prompt_passes"),
                "rollout_token_coverage": budget.get(
                    "rollout_token_coverage"
                ),
            },
            "both values finite and positive",
        ),
    ]
    checks.extend(
        _retention_check(report, parent, name)
        for name in (
            "instruction.success",
            "format.success",
            "qa.success",
            "multiturn.success",
            "preference.accuracy",
        )
    )
    return _report("grpo", checks)


def select_stage_winner(
    stage: str,
    candidates: list[tuple[str | Path, str | Path, str | Path]],
) -> dict[str, Any]:
    """Select only among passing candidates using the pre-registered ordering."""

    if stage not in {"sft", "dpo", "grpo"}:
        raise ContractError("winner selection stage must be sft, dpo, or grpo")
    ranked = []
    for run_value, report_value, gate_value in candidates:
        run = Path(run_value)
        report = _read_json(report_value)
        gate = _read_json(gate_value)
        if gate.get("stage") != stage:
            raise ContractError(
                f"candidate gate is not for stage {stage}: {gate_value}"
            )
        if gate.get("ok") is not True:
            continue
        _, final = _run_metrics(run)
        if stage == "sft":
            task_successes = sum(
                _success_count(report, name)[0]
                for name in (
                    "instruction.success",
                    "format.success",
                    "qa.success",
                    "verifiable.reward",
                )
            )
            score = (
                float(task_successes),
                -_metric_value(report, "corpus.bpb.all"),
                -_number(final.get("eval_loss"), "eval_loss"),
            )
        elif stage == "dpo":
            preference_successes = _success_count(
                report,
                "preference.accuracy",
            )[0]
            score = (
                float(preference_successes),
                -_metric_value(report, "corpus.bpb.all"),
                -_number(final.get("eval_loss"), "eval_loss"),
            )
        else:
            score = (
                _number(final.get("eval_reward_mean"), "eval_reward_mean"),
                -_number(final.get("eval_approx_kl"), "eval_approx_kl"),
                -_metric_value(report, "corpus.bpb.all"),
            )
        result = _read_json(run / "training_result.json")
        ranked.append(
            {
                "run_dir": str(run),
                "capability_report": str(report_value),
                "gate_report": str(gate_value),
                "final_checkpoint": str(result["final_checkpoint"]),
                "score": list(score),
            }
        )
    if not ranked:
        raise ContractError(f"no {stage} candidate passed its stage gate")
    ranked.sort(
        key=lambda candidate: (
            tuple(-value for value in candidate["score"]),
            candidate["run_dir"],
        )
    )
    return {
        "schema_version": "1.0",
        "stage": stage,
        "selection_rule": {
            "sft": "task-successes-desc,bpb-asc,dev-loss-asc",
            "dpo": "preference-successes-desc,bpb-asc,dev-loss-asc",
            "grpo": "dev-reward-desc,dev-kl-asc,bpb-asc",
        }[stage],
        "winner": ranked[0],
        "passing_candidates": ranked,
    }


def _run_metrics(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = Path(run_dir)
    rows = []
    try:
        with (run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read run metrics: {exc}") from exc
    baseline = next(
        (row for row in rows if row.get("event") == "baseline"),
        None,
    )
    final = next(
        (row for row in reversed(rows) if "eval_loss" in row),
        None,
    )
    if not isinstance(baseline, dict) or not isinstance(final, dict):
        raise ContractError("run metrics require baseline and final evaluation rows")
    return baseline, final


def _metric(report: dict[str, Any], name: str) -> dict[str, Any]:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get(name), dict):
        raise ContractError(f"capability report is missing metric {name}")
    metric = metrics[name]
    if metric.get("status") != "ok":
        raise ContractError(f"capability metric {name} is not comparable")
    return metric


def _metric_value(report: dict[str, Any], name: str) -> float:
    return _number(_metric(report, name).get("value"), name)


def _success_count(report: dict[str, Any], name: str) -> tuple[int, int]:
    metric = _metric(report, name)
    count = int(metric.get("count", 0))
    value = _number(metric.get("value"), name)
    return int(round(value * count)), count


def _minimum_successes(
    report: dict[str, Any],
    name: str,
    minimum: int,
) -> dict[str, Any]:
    successes, count = _success_count(report, name)
    return _check(
        name,
        successes >= minimum,
        {"successes": successes, "count": count},
        {"minimum_successes": minimum},
    )


def _success_delta(
    check_name: str,
    report: dict[str, Any],
    parent: dict[str, Any],
    metric_name: str,
    *,
    minimum_delta: int,
) -> dict[str, Any]:
    current, count = _success_count(report, metric_name)
    previous, previous_count = _success_count(parent, metric_name)
    if count != previous_count:
        raise ContractError(f"capability denominator changed for {metric_name}")
    delta = current - previous
    return _check(
        check_name,
        delta >= minimum_delta,
        {"delta": delta, "current": current, "parent": previous, "count": count},
        {"minimum_delta": minimum_delta},
    )


def _retention_check(
    report: dict[str, Any],
    parent: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    current, count = _success_count(report, name)
    previous, previous_count = _success_count(parent, name)
    if count != previous_count:
        raise ContractError(f"capability denominator changed for {name}")
    return _check(
        f"{name}-retention",
        current >= previous - 1,
        {"current": current, "parent": previous, "count": count},
        {"maximum_case_drop": 1},
    )


def _relative_decrease(
    name: str,
    actual: Any,
    baseline: Any,
    *,
    minimum_fraction: float,
) -> dict[str, Any]:
    actual_value = _number(actual, name)
    baseline_value = _number(baseline, f"{name} baseline")
    maximum = baseline_value * (1 - minimum_fraction)
    return _check(name, actual_value <= maximum, actual_value, {"maximum": maximum})


def _maximum_ratio(
    name: str,
    actual: float,
    baseline: float,
    maximum_ratio: float,
) -> dict[str, Any]:
    return _check(
        name,
        actual <= baseline * maximum_ratio,
        actual,
        {"baseline": baseline, "maximum_ratio": maximum_ratio},
    )


def _bounded(
    name: str,
    actual: Any,
    *,
    minimum: float,
    maximum: float,
) -> dict[str, Any]:
    value = _number(actual, name)
    return _check(
        name,
        minimum <= value <= maximum,
        value,
        {"minimum": minimum, "maximum": maximum},
    )


def _maximum(name: str, actual: Any, maximum: float) -> dict[str, Any]:
    value = _number(actual, name)
    return _check(name, value <= maximum, value, {"maximum": maximum})


def _number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ContractError(f"{name} must be finite")
    return number


def _positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _wilson_lower(successes: int, count: int) -> float:
    if count <= 0 or not 0 <= successes <= count:
        raise ContractError("invalid binomial counts")
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    center = proportion + z * z / (2 * count)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / count + z * z / (4 * count * count)
    )
    return (center - margin) / denominator


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _evaluation_metrics(value: dict[str, Any]) -> dict[str, Any]:
    metrics = value.get("metrics")
    return dict(metrics) if isinstance(metrics, dict) else value


def _check(name: str, ok: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "actual": actual,
        "expected": expected,
    }


def _report(stage: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "stage": stage,
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
    }
