"""Comparable, immutable evaluation reports with explicit baseline deltas."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from llm_lifecycle_lab.artifacts import RunArtifacts
from llm_lifecycle_lab.contracts import utc_now
from llm_lifecycle_lab.evaluation.suite import digest, read_json
from llm_lifecycle_lab.exceptions import ArtifactError, ContractError


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != "1.0":
        raise ContractError("unsupported capability report schema")
    if report.get("protocol_sha256") != digest(report["protocol"]):
        raise ContractError("capability report protocol hash mismatch")
    if not report.get("metrics"):
        raise ContractError("capability report contains no metrics")


def compare_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ContractError("comparison requires at least two reports in stage order")
    for report in reports:
        validate_report(report)
    first = reports[0]
    for report in reports[1:]:
        if report["protocol_sha256"] != first["protocol_sha256"]:
            changed = sorted(
                key
                for key in first["protocol"].keys() | report["protocol"].keys()
                if first["protocol"].get(key) != report["protocol"].get(key)
            )
            raise ContractError(
                "incomparable evaluation protocols: " + ", ".join(changed)
            )
        if report["metrics"].keys() != first["metrics"].keys():
            raise ContractError("incomparable metric sets")
    rows = {}
    for name, initial in first["metrics"].items():
        values = [report["metrics"][name] for report in reports]
        if any(
            v["count"] != initial["count"]
            or v["direction"] != initial["direction"]
            or v["baseline"] != initial["baseline"]
            for v in values
        ):
            raise ContractError(f"incomparable metric denominator/baseline: {name}")
        comparable = all(v["status"] == "ok" and v["value"] is not None for v in values)
        rows[name] = {
            "values": [value["value"] for value in values],
            "count": initial["count"],
            "direction": initial["direction"],
            "baseline": initial["baseline"],
            "delta_from_first": [
                value["value"] - initial["value"] if comparable else None
                for value in values
            ],
            "status": "ok" if comparable else "not-comparable",
        }
    return {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "protocol_sha256": first["protocol_sha256"],
        "models": [report["model"] for report in reports],
        "metrics": rows,
        "interpretation": (
            "Delta is current minus first. Negative BPB/repetition delta is better; "
            "positive task/reward delta is better. No aggregate capability score."
        ),
    }


def _number(value: float | None) -> str:
    if value is None:
        return "未检查"
    return f"{value:.4e}" if 0 < abs(value) < 1e-4 else f"{value:.6f}"


def render_report(report: dict[str, Any]) -> str:
    model = report["model"]
    lines = [
        "# 能力评测成绩单",
        "",
        f"- 模型：`{model['run_id']}/{model['checkpoint_id']}`",
        f"- 阶段：`{model['stage']}`",
        f"- 权重 SHA-256：`{model['weights_sha256']}`",
        f"- 协议 SHA-256：`{report['protocol_sha256']}`",
        "- 小型诊断探针，不是通用能力排行榜；不能用少量样例推断总体能力。",
        "",
        "| 指标 | 数值 | 分母 | 基线 | 方向 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, value in report["metrics"].items():
        number = _number(value["value"])
        lines.append(
            f"| {name} | {number} | {value['count']} | "
            f"{_number(value['baseline']['value'])} ({value['baseline']['kind']}) | "
            f"{value['direction']} |"
        )
    lines.extend(["", "## 口径与边界", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", "## 固定续写", ""])
    for sample in report["samples"]:
        if sample["kind"] == "continuation":
            # JSON report retains exact raw output; escape Markdown table fences.
            text = sample["output"]["text"].replace("`", "'").replace("\n", " ")
            lines.append(f"- `{sample['prompt']}` → {text}")
    if report.get("comparison"):
        lines.extend(
            ["", "## 相对阶段前基线", "", "| 指标 | 当前减基线 |", "| --- | ---: |"]
        )
        for name, row in report["comparison"]["metrics"].items():
            delta = row["delta_from_first"][-1]
            shown = "不可比较" if delta is None else f"{delta:+.6f}"
            lines.append(f"| {name} | {shown} |")
    return "\n".join(lines) + "\n"


def write_report(directory: str | Path, report: dict[str, Any], markdown: str) -> Path:
    target = Path(directory)
    if target.exists():
        raise ArtifactError(f"report directory exists; refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))
    try:
        artifacts = RunArtifacts(run_id=target.name, path=temporary)
        artifacts.write_json("report.json", report)
        artifacts.write_text("report.md", markdown)
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def compare_files(paths: list[Path], output: Path) -> dict[str, Any]:
    reports = [read_json(path) for path in paths]
    comparison = compare_reports(reports)
    labels = [f"{r['model']['stage']}:{r['model']['run_id']}" for r in reports]
    lines = [
        "# 纵向能力成绩单",
        "",
        "| 指标 | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    for name, row in comparison["metrics"].items():
        cells = ["未检查" if v is None else f"{v:.6f}" for v in row["values"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend(["", comparison["interpretation"], ""])
    write_report(output, comparison, "\n".join(lines))
    return comparison
