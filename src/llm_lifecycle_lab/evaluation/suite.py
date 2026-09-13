"""Small, explicit evaluation contracts and deterministic response scorers."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.config import canonical_json, load_mapping
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.tokenizer.native import SPECIAL_TOKENS

SCORING_VERSION = "lifecycle-rules-v1"
KINDS = {
    "corpus",
    "continuation",
    "instruction",
    "qa",
    "format",
    "multiturn",
    "preference",
}


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object: {path}")
    return value


def load_suite(path: str | Path) -> dict[str, Any]:
    suite = dict(load_mapping(path))
    if suite.get("schema_version") != "1.0":
        raise ContractError("unsupported evaluation suite schema_version")
    for field in ("suite_id", "license", "description"):
        require_text(suite.get(field), field)
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ContractError("suite.cases must be a nonempty list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ContractError("each evaluation case must be an object")
        for field in ("id", "group", "language", "kind"):
            require_text(case.get(field), field)
        if case["id"] in seen:
            raise ContractError(f"duplicate evaluation id: {case['id']}")
        seen.add(case["id"])
        if case["language"] not in {"en", "zh"} or case["kind"] not in KINDS:
            raise ContractError(f"unsupported language/kind: {case['id']}")
        kind = case["kind"]
        if kind == "corpus":
            require_text(case.get("text"), "text")
        elif kind == "multiturn":
            turns = case.get("turns")
            if not isinstance(turns, list) or len(turns) < 2:
                raise ContractError("multiturn requires at least two turns")
            for turn in turns:
                require_text(turn.get("prompt"), "turn.prompt")
                validate_rule(turn.get("rule"))
        else:
            require_text(case.get("prompt"), "prompt")
            if kind == "preference":
                require_text(case.get("chosen"), "chosen")
                require_text(case.get("rejected"), "rejected")
                if case["chosen"] == case["rejected"]:
                    raise ContractError("preference alternatives must differ")
            elif kind != "continuation":
                validate_rule(case.get("rule"))
    if {case["language"] for case in cases} != {"en", "zh"}:
        raise ContractError("capability suite must cover both en and zh")
    # Enforce a strict JSON representation, including finite numbers.
    digest(suite)
    return suite


def require_text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be nonempty text")
    if any(token in value for token in SPECIAL_TOKENS):
        raise ContractError(f"{field} contains a Native control token")


def validate_rule(rule: Any) -> None:
    if not isinstance(rule, dict):
        raise ContractError("task requires a scoring rule")
    if rule.get("type") == "exact":
        require_text(rule.get("answer"), "rule.answer")
        if "choices" in rule:
            choices = rule["choices"]
            if (
                not isinstance(choices, list)
                or len(choices) < 2
                or any(not isinstance(x, str) or not x for x in choices)
                or len(set(choices)) != len(choices)
                or rule["answer"] not in choices
            ):
                raise ContractError("invalid rule choices")
    elif rule.get("type") == "json":
        if not isinstance(rule.get("answer"), dict) or not rule["answer"]:
            raise ContractError("JSON rule requires a nonempty object answer")
        digest(rule["answer"])
    else:
        raise ContractError("rule.type must be exact or json")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON float")
    return number


def score_response(text: str, rule: dict[str, Any]) -> float:
    validate_rule(rule)
    if rule["type"] == "exact":
        # Only trim outer whitespace. Extra prose, changed case and control tokens fail.
        return float(text.strip() == rule["answer"])
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (ValueError, TypeError, RecursionError):
        return 0.0
    # JSON types matter: true is not 1, and no extra keys are permitted.
    return float(canonical_json(value) == canonical_json(rule["answer"]))


def rule_baseline(rule: dict[str, Any]) -> dict[str, Any]:
    if "choices" in rule:
        return {
            "kind": "analytic-random",
            "value": 1.0 / len(rule["choices"]),
            "description": "Uniform selection from the declared answer choices.",
        }
    return {
        "kind": "rule",
        "value": score_response("", rule),
        "description": "Always-empty response; a floor, not a competent solver.",
    }


def repeated_ngram_fraction(ids: list[int], n: int = 3) -> float:
    grams = Counter(tuple(ids[i : i + n]) for i in range(len(ids) - n + 1))
    count = sum(grams.values())
    return (count - len(grams)) / count if count else 0.0


def metric(
    value: float | None,
    *,
    count: int,
    direction: str,
    baseline: dict[str, Any],
    status: str = "ok",
) -> dict[str, Any]:
    if value is not None and not math.isfinite(value):
        raise ContractError("evaluation metric must be finite")
    if not math.isfinite(baseline["value"]) or count < 0:
        raise ContractError("invalid metric baseline/count")
    return {
        "value": value,
        "count": count,
        "direction": direction,
        "status": status,
        "baseline": baseline,
    }
