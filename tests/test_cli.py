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

    def test_missing_file_prints_clean_error(self):
        proc = run_cli("run", "tasks/nope.json", "solutions/add.c")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("error:", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_attempt_below_one_rejected(self):
        proc = run_cli("run", "tasks/001_add.json", "solutions/add.c", "--attempt", "0")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--attempt must be >= 1", proc.stderr)


class CliMarlParserTest(unittest.TestCase):
    def test_marl_arguments_parse(self):
        from arena.cli import build_parser

        args = build_parser().parse_args([
            "marl", "--episodes", "9", "--attempts", "2", "--alpha", "0.4",
            "--gamma", "0.8", "--epsilon", "0.3", "--seed", "42",
            "--state", "marl_state.json", "--log", "trajectories/marl.jsonl",
        ])
        self.assertEqual(args.command, "marl")
        self.assertEqual(args.episodes, 9)
        self.assertEqual(args.attempts, 2)
        self.assertEqual(args.alpha, 0.4)
        self.assertEqual(args.gamma, 0.8)
        self.assertEqual(args.epsilon, 0.3)
        self.assertEqual(args.seed, 42)
        self.assertEqual(str(args.state), "marl_state.json")
        self.assertEqual(str(args.log), "trajectories/marl.jsonl")

    def test_marl_defaults(self):
        from arena.cli import build_parser

        args = build_parser().parse_args(["marl"])
        self.assertEqual(args.episodes, 9)
        self.assertEqual(args.attempts, 2)
        self.assertEqual(args.seed, 42)

    def test_marl_cli_prints_summary_and_writes_report_without_model(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from arena.cli import main
        from test_marl import fake_generate

        with tempfile.TemporaryDirectory() as td:
            state = f"{td}/state.json"
            log = f"{td}/episodes.jsonl"
            report = f"{td}/report.md"
            output = StringIO()
            with patch("arena.solver_llm.load_model", return_value=(None, None)), \
                 patch("arena.solver_llm.llm_generate",
                       side_effect=lambda model, tokenizer, messages: fake_generate(messages)), \
                 redirect_stdout(output):
                code = main(["marl", "--episodes", "2", "--attempts", "1",
                             "--state", state, "--log", log, "--report", report])
            self.assertEqual(code, 0)
            self.assertIn("MARL RUN COMPLETE", output.getvalue())
            self.assertIn("Task selection:", output.getvalue())
            self.assertNotIn("Q_challenger:", output.getvalue())
            with open(report, encoding="utf-8") as f:
                body = f.read()
            self.assertIn("Episodes this run: 2", body)
            self.assertIn("Final learned policy by state", body)
            self.assertIn(log, body)


if __name__ == "__main__":
    unittest.main()
