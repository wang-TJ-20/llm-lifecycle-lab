from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_lifecycle_lab.gates import (
    check_base_gate,
    check_dpo_gate,
    check_grpo_gate,
    check_sft_gate,
    select_stage_winner,
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _metric(value: float, count: int) -> dict:
    return {"value": value, "count": count, "status": "ok"}


def _capability(*, preference: int = 10, bpb: float = 1.0) -> dict:
    counts = {
        "instruction.success": (8, 32),
        "format.success": (6, 24),
        "qa.success": (8, 32),
        "multiturn.success": (4, 16),
        "verifiable.reward": (20, 112),
        "preference.accuracy": (preference, 32),
    }
    metrics = {
        name: _metric(successes / count, count)
        for name, (successes, count) in counts.items()
    }
    for name in ("instruction.success", "format.success", "qa.success"):
        successes, count = counts[name]
        metrics[f"{name}.en"] = _metric((successes // 2) / (count // 2), count // 2)
        metrics[f"{name}.zh"] = _metric((successes // 2) / (count // 2), count // 2)
    metrics["corpus.bpb.all"] = _metric(bpb, 100)
    return {"metrics": metrics}


class StageGateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _run(self, baseline: dict, final: dict) -> Path:
        run = self.root / f"run-{len(list(self.root.glob('run-*')))}"
        run.mkdir()
        (run / "metrics.jsonl").write_text(
            json.dumps({"event": "baseline", **baseline})
            + "\n"
            + json.dumps(final)
            + "\n",
            encoding="utf-8",
        )
        _write(
            run / "training_result.json",
            {"final_checkpoint": str(run / "checkpoints/final")},
        )
        return run

    def test_base_gate_accepts_nested_frozen_test_report(self) -> None:
        run = self.root / "base"
        _write(run / "training_result.json", {"target_token_coverage": 1.0})
        _write(run / "runtime_environment.json", {"code": {"dirty": False}})
        _write(run / "run_manifest.json", {"status": "completed"})
        dev = _write(
            self.root / "dev.json",
            {"eval_loss": 2.8, "eval_en_loss": 1.9, "eval_zh_loss": 3.7},
        )
        test = _write(
            self.root / "test.json",
            {"eval_loss": 3.0, "eval_en_loss": 2.1, "eval_zh_loss": 3.9},
        )
        baseline = _write(
            self.root / "baseline.json",
            {
                "metrics": {
                    "eval_loss": 3.04,
                    "eval_en_loss": 2.15,
                    "eval_zh_loss": 3.94,
                }
            },
        )
        self.assertTrue(
            check_base_gate(
                run,
                dev_evaluation=dev,
                test_evaluation=test,
                baseline_test_evaluation=baseline,
            )["ok"]
        )

    def test_posttraining_gates_accept_qualified_evidence(self) -> None:
        parent = _write(self.root / "parent.json", _capability())
        current = _write(self.root / "current.json", _capability(preference=13))
        data = _write(
            self.root / "data.json",
            {
                "ok": True,
                "stage_source_id_overlap": {
                    "sft-dpo": 0,
                    "sft-grpo": 0,
                    "dpo-grpo": 0,
                },
            },
        )

        sft = self._run({"eval_loss": 2.0}, {"eval_loss": 1.8})
        self.assertTrue(
            check_sft_gate(
                sft,
                capability_report=current,
                parent_capability_report=parent,
                data_check=data,
            )["ok"]
        )

        dpo = self._run(
            {"eval_loss": 0.7},
            {
                "eval_loss": 0.6,
                "eval_pair_accuracy": 0.6,
                "eval_normalization_count": 400,
            },
        )
        self.assertTrue(
            check_dpo_gate(
                dpo,
                capability_report=current,
                parent_capability_report=parent,
            )["ok"]
        )

        grpo = self._run(
            {"eval_loss": 1.0, "eval_reward_mean": 0.10},
            {
                "eval_loss": 0.9,
                "eval_reward_mean": 0.16,
                "eval_zero_variance_groups": 0.5,
                "eval_approx_kl": 0.08,
            },
        )
        _write(
            grpo / "grpo_budget_summary.json",
            {"prompt_passes": 1.0, "rollout_token_coverage": 0.8},
        )
        self.assertTrue(
            check_grpo_gate(
                grpo,
                capability_report=current,
                parent_capability_report=parent,
            )["ok"]
        )

        gate = _write(
            self.root / "sft-gate.json",
            {"stage": "sft", "ok": True},
        )
        selection = select_stage_winner(
            "sft",
            [(sft, current, gate)],
        )
        self.assertEqual(selection["winner"]["run_dir"], str(sft))


if __name__ == "__main__":
    unittest.main()
