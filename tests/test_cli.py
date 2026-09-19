"""End-to-end CLI tests: python -m arena run <task.json> <solution.c>."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HAS_TCC = shutil.which("tcc") is not None


def run_cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(shutil.which("tcc") or "") + os.pathsep + env.get("PATH", "")
    return subprocess.run(
        [sys.executable, "-m", "arena", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


@unittest.skipUnless(HAS_TCC, "tcc not on PATH")
class CliRunTest(unittest.TestCase):
    def test_correct_solution_passes(self):
        proc = run_cli("run", "tasks/001_add.json", "solutions/add.c")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PASS", proc.stdout)
        self.assertIn("4/4 passed", proc.stdout)
        self.assertIn("reward:  16.0", proc.stdout)

    def test_wrong_solution_fails_with_diagnostics(self):
        proc = run_cli("run", "tasks/001_add.json", "solutions/wrong.c")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("FAIL", proc.stdout)
        self.assertIn("expected:", proc.stdout)
        self.assertIn("got:", proc.stdout)
        self.assertNotIn("4/4 passed", proc.stdout)

    def test_invalid_c_returns_compiler_diagnostics(self):
        proc = run_cli("run", "tasks/001_add.json", "solutions/bad.c")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("COMPILE ERROR", proc.stdout)
        self.assertIn("compiler stderr:", proc.stdout)

    def test_infinite_loop_reports_timeout(self):
        proc = run_cli("run", "tasks/001_add.json", "solutions/infinite.c")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("timed out", proc.stdout)

    def test_max_array_task_end_to_end(self):
        proc = run_cli("run", "tasks/002_max_array.json", "solutions/max_array.c")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("5/5 passed", proc.stdout)

    def test_reverse_string_task_end_to_end(self):
        proc = run_cli("run", "tasks/003_reverse_string.json", "solutions/reverse.c")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("4/4 passed", proc.stdout)

    def test_json_flag_prints_structured_result(self):
        proc = run_cli("run", "--json", "tasks/001_add.json", "solutions/add.c")
        self.assertEqual(proc.returncode, 0)
        data = json.loads(proc.stdout)
        self.assertTrue(data["compiled"])
        self.assertEqual(data["passed"], 4)
        self.assertEqual(data["total"], 4)
        self.assertEqual(data["reward"], 16.0)
        self.assertEqual(len(data["tests"]), 4)

    def test_log_flag_appends_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = os.path.join(td, "traj.jsonl")
            proc_pass = run_cli("run", "tasks/001_add.json", "solutions/add.c", "--log", log_path, "--attempt", "1")
            self.assertEqual(proc_pass.returncode, 0)
            proc_retry = run_cli("run", "tasks/001_add.json", "solutions/add.c", "--log", log_path, "--attempt", "2")
            self.assertEqual(proc_retry.returncode, 0)
            with open(log_path, encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            second = json.loads(lines[1])
            self.assertEqual(first["task_id"], "001")
            self.assertTrue(first["success"])
            self.assertEqual(first["reward"] - second["reward"], 0.5)


if __name__ == "__main__":
    unittest.main()