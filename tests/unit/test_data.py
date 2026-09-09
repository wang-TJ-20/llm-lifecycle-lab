from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_lifecycle_lab.contracts import RecordKind
from llm_lifecycle_lab.data import (
    SplitRatios,
    prepare_dataset,
    split_records,
    validate_jsonl,
    verify_data_manifest,
)
from llm_lifecycle_lab.exceptions import DataValidationError


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records),
        encoding="utf-8",
    )


class DataValidationTests(unittest.TestCase):
    def test_sft_validation_accepts_valid_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sft.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "id": "s1",
                        "messages": [
                            {"role": "user", "content": "2+2?"},
                            {"role": "assistant", "content": "4"},
                        ],
                    }
                ],
            )

            result = validate_jsonl(path, RecordKind.SFT)

            self.assertTrue(result.report.ok)
            self.assertEqual(result.report.valid_records, 1)

    def test_dpo_validation_rejects_identical_pair_and_duplicate_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dpo.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "id": "d1",
                        "prompt": "question",
                        "chosen": "same",
                        "rejected": "same",
                    },
                    {
                        "id": "d1",
                        "prompt": "question",
                        "chosen": "better",
                        "rejected": "worse",
                    },
                ],
            )

            result = validate_jsonl(path, RecordKind.DPO)
            codes = {issue.code for issue in result.report.issues}

            self.assertFalse(result.report.ok)
            self.assertEqual(codes, {"duplicate_id", "identical_pair"})

    def test_non_finite_json_number_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pretrain.jsonl"
            path.write_text(
                '{"id":"p1","text":"hello","source":"test","score":NaN}\n',
                encoding="utf-8",
            )

            result = validate_jsonl(path, RecordKind.PRETRAIN)

            self.assertEqual(result.report.issues[0].code, "invalid_json")


class SplitTests(unittest.TestCase):
    def test_group_assignment_is_independent_of_input_order(self) -> None:
        records = [
            {"id": f"id-{index}", "source_id": f"group-{index // 2}"}
            for index in range(30)
        ]
        ratios = SplitRatios()

        first = split_records(
            records,
            ratios=ratios,
            seed=7,
            group_by="source_id",
        )
        second = split_records(
            list(reversed(records)),
            ratios=ratios,
            seed=7,
            group_by="source_id",
        )

        def assignments(
            splits: dict[str, list[dict[str, object]]],
        ) -> dict[str, str]:
            return {
                str(record["id"]): split_name
                for split_name, rows in splits.items()
                for record in rows
            }

        self.assertEqual(assignments(first), assignments(second))
        for rows in first.values():
            groups = {str(record["source_id"]) for record in rows}
            for other_rows in first.values():
                if rows is other_rows:
                    continue
                self.assertTrue(
                    groups.isdisjoint(
                        {str(record["source_id"]) for record in other_rows}
                    )
                )

    def test_prepare_and_verify_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pretrain.jsonl"
            _write_jsonl(
                source,
                [
                    {
                        "id": f"p{index}",
                        "text": f"sample {index}",
                        "source": "fixture",
                    }
                    for index in range(20)
                ],
            )
            output = root / "prepared"
            manifest = prepare_dataset(
                source,
                output,
                dataset_id="fixture-v1",
                record_kind=RecordKind.PRETRAIN,
                license_name="test-only",
                seed=42,
            )

            self.assertEqual(sum(split.records for split in manifest.splits), 20)
            self.assertEqual(
                verify_data_manifest(output / "data_manifest.json"),
                [],
            )

            with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"id":"tampered"}\n')

            failures = verify_data_manifest(output / "data_manifest.json")
            self.assertTrue(any("hash mismatch" in item for item in failures))
            self.assertTrue(any("record count mismatch" in item for item in failures))

    def test_prepare_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pretrain.jsonl"
            _write_jsonl(
                source,
                [{"id": "p1", "text": "sample", "source": "fixture"}],
            )
            output = root / "prepared"
            output.mkdir()

            with self.assertRaisesRegex(
                DataValidationError,
                "refusing to overwrite",
            ):
                prepare_dataset(
                    source,
                    output,
                    dataset_id="fixture-v1",
                    record_kind=RecordKind.PRETRAIN,
                    license_name="test-only",
                )


if __name__ == "__main__":
    unittest.main()
