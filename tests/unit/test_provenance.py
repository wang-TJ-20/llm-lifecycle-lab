from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_lifecycle_lab.provenance import source_tree_sha256


class ProvenanceTests(unittest.TestCase):
    def test_source_hash_tracks_contents_not_checkout_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("first", "second"):
                source = root / name / "src/llm_lifecycle_lab"
                source.mkdir(parents=True)
                (source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
            expected = source_tree_sha256(root / "first")
            self.assertEqual(len(expected), 64)
            self.assertEqual(source_tree_sha256(root / "second"), expected)
            (root / "second/README.md").write_text("notes", encoding="utf-8")
            self.assertEqual(source_tree_sha256(root / "second"), expected)
            (root / "second/src/llm_lifecycle_lab/model.py").write_text(
                "VALUE = 2\n", encoding="utf-8"
            )
            self.assertNotEqual(source_tree_sha256(root / "second"), expected)

    def test_missing_source_has_no_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(source_tree_sha256(directory))
