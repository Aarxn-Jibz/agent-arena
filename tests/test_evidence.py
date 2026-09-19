import json
import tempfile
import unittest
from pathlib import Path

from arena.evidence import sha256_text, write_episode


def sample_record():
    return {
        "run_id": "demo", "episode_id": 1, "benchmark": "example",
        "seeds": {"challenge": 7, "inputs": 8}, "git_before": "abc123",
        "challenger": {"request": {"goal": "follow specification"}, "rationale": "probe edge cases"},
        "solver": {"response": "candidate", "candidate": "int main(){return 0;}",
                   "patch": "+int main(){return 0;}"},
        "input_generation": {"generator": "example-v1", "seed": 8, "config": {}},
        "build": {"exit_code": 0, "stdout": "", "stderr": ""},
        "correctness": {"passed": 1, "total": 1},
        "performance": {"elapsed_ms": 2},
        "judge": {"feedback": "accepted", "resource_usage": {"memory_bytes": 1000}},
        "rewards": {"solver": 1.0, "challenger": 0.0},
        "outcome": "accepted", "git_after": "def456",
        "timestamps": {"started_at": "2026-01-01T00:00:00+00:00",
                       "candidate_at": "2026-01-01T00:00:01+00:00",
                       "finished_at": "2026-01-01T00:00:02+00:00"},
    }


class EvidenceTest(unittest.TestCase):
    def test_json_markdown_and_viewer_events(self):
        with tempfile.TemporaryDirectory() as td:
            json_path, md_path = write_episode(td, sample_record())
            saved = json.loads(json_path.read_text())
            self.assertEqual(saved["schema_version"], 1)
            self.assertEqual(saved["hashes"]["candidate_sha256"],
                             sha256_text(sample_record()["solver"]["candidate"]))
            self.assertIn("## Challenger", md_path.read_text())
            events = [json.loads(x) for x in (Path(td) / "demo/events.jsonl").read_text().splitlines()]
            self.assertEqual([e["actor"] for e in events], ["challenger", "solver", "judge"])
            self.assertEqual(events[-1]["outcome"], "accepted")
            with self.assertRaises(FileExistsError):
                write_episode(td, sample_record())

    def test_rejects_invalid_lineage_and_paths(self):
        with tempfile.TemporaryDirectory() as td:
            record = sample_record()
            record["run_id"] = "../escape"
            with self.assertRaises(ValueError):
                write_episode(td, record)
            record = sample_record()
            record["outcome"] = "rejected"
            with self.assertRaises(ValueError):
                write_episode(td, record)
