"""Tests for JSONL trajectory logging."""

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import asdict

from arena.evaluate import evaluate
from arena.log import append_trajectory, trajectory_entry

HAS_TCC = shutil.which("tcc") is not None

ADD_TASK = {
    "id": "001",
    "title": "Add two integers",
    "prompt": "Read two integers and print their sum.",
    "timeout_seconds": 2,
    "tests": [{"input": "2 3\n", "expected_output": "5\n"}],
}
ADD_CORRECT = '#include <stdio.h>\nint main(void){int a,b;scanf("%d %d",&a,&b);printf("%d\\n",a+b);return 0;}\n'


class TrajectoryEntryTest(unittest.TestCase):
    def _result_dict(self):
        if not HAS_TCC:
            self.skipTest("tcc not on PATH")
        return asdict(evaluate(ADD_CORRECT, ADD_TASK))

    def test_entry_has_all_fields(self):
        result = self._result_dict()
        entry = trajectory_entry(ADD_TASK, 3, ADD_CORRECT, result)
        self.assertEqual(entry["task_id"], "001")
        self.assertEqual(entry["attempt"], 3)
        self.assertEqual(entry["source"], ADD_CORRECT)
        self.assertEqual(entry["reward"], result["reward"])
        self.assertEqual(entry["success"], result["success"])
        self.assertEqual(entry["result"], result)
        # ISO timestamp present
        self.assertRegex(entry["timestamp"], r"^\d{4}-\d{2}-\d{2}T")

    def test_entry_is_json_serializable(self):
        result = self._result_dict()
        entry = trajectory_entry(ADD_TASK, 1, ADD_CORRECT, result)
        json.dumps(entry)  # must not raise


class AppendTrajectoryTest(unittest.TestCase):
    def _entry(self, attempt=1):
        if not HAS_TCC:
            self.skipTest("tcc not on PATH")
        result = asdict(evaluate(ADD_CORRECT, ADD_TASK))
        return trajectory_entry(ADD_TASK, attempt, ADD_CORRECT, result)

    def test_appends_valid_jsonl_and_creates_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = os.path.join(td, "deep", "dir", "trajectories.jsonl")
            append_trajectory(log_path, self._entry(attempt=1))
            append_trajectory(log_path, self._entry(attempt=2))
            with open(log_path, encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            self.assertEqual(len(lines), 2)
            parsed = [json.loads(line) for line in lines]
            self.assertEqual([p["attempt"] for p in parsed], [1, 2])
            # every line is one JSON object, source survives round-trip exactly
            self.assertEqual(parsed[0]["source"], ADD_CORRECT)
            self.assertTrue(parsed[1]["success"])

    def test_log_does_not_mix_entries_into_one_line(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = os.path.join(td, "trajectories.jsonl")
            append_trajectory(log_path, self._entry())
            append_trajectory(log_path, self._entry())
            with open(log_path, encoding="utf-8") as f:
                raw = f.read()
            if raw:
                self.assertNotIn("\n\n", raw.replace("\r\n", "\n"))
            self.assertEqual(raw.count("\n"), 2)


if __name__ == "__main__":
    unittest.main()