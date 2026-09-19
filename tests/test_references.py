import tempfile
import unittest

from arena.evidence import write_episode
from test_evidence import sample_record


class ReferenceEvidenceTest(unittest.TestCase):
    def test_reference_read_is_retained_in_solver_event(self):
        record = sample_record()
        record["reference_reads"] = [{"document": "references/c-language.md",
                                      "section": "Pointers and lifetime",
                                      "read_at": "2026-01-01T00:00:00+00:00"}]
        with tempfile.TemporaryDirectory() as td:
            path, _ = write_episode(td, record)
            self.assertIn("reference_reads", path.read_text())
            with open(path.parent / "events.jsonl", encoding="utf-8") as stream:
                stream.readline()
                self.assertIn("references/c-language.md", stream.readline())
