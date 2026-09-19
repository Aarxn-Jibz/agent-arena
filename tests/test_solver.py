"""Tests for the model-agnostic Solver retry loop.

A fake `generate` callable stands in for the LLM, so these tests never touch
transformers/torch or download anything.
"""

import json
import shutil
import tempfile
import unittest

from arena.solver import build_feedback, build_prompt, extract_c_code, solve

HAS_TCC = shutil.which("tcc") is not None

ADD_TASK = {
    "id": "001",
    "title": "Add two integers",
    "prompt": "Read two integers a and b from stdin and print their sum.",
    "timeout_seconds": 2,
    "tests": [{"input": "2 3\n", "expected_output": "5\n"}],
}

GOOD_C = '#include <stdio.h>\nint main(void){int a,b;scanf("%d %d",&a,&b);printf("%d\\n",a+b);return 0;}\n'
BAD_C = "int main(void){ this is not C }\n"


class ExtractCTest(unittest.TestCase):
    def test_fenced_block(self):
        self.assertEqual(extract_c_code("```c\nint main(){}\n```"), "int main(){}\n")

    def test_fenced_block_without_language(self):
        self.assertEqual(extract_c_code("```\nint main(){}\n```"), "int main(){}\n")

    def test_no_fence_returns_none(self):
        self.assertIsNone(extract_c_code("int main(){}"))

    def test_empty_block_returns_none(self):
        self.assertIsNone(extract_c_code("```c\n\n```"))


class BuildMessagesTest(unittest.TestCase):
    def test_prompt_hides_expected_outputs(self):
        prompt = build_prompt(ADD_TASK)
        self.assertIn("Read two integers a and b", prompt)
        self.assertIn("id=001", prompt)
        self.assertNotIn("expected_output", prompt)
        self.assertNotIn("5", prompt)
        self.assertIn("```c", prompt)


@unittest.skipUnless(HAS_TCC, "tcc not on PATH")
class SolveLoopTest(unittest.TestCase):
    def test_fixes_compile_error_on_retry(self):
        def generate(messages):
            # first reply has no code fence at all; second is correct
            if messages[-1]["role"] == "user" and "Attempt 1 failed" in messages[-1]["content"]:
                return "```c\n" + GOOD_C + "```"
            return "I will write C:\n```c\n" + BAD_C + "\n```"

        result = solve(ADD_TASK, generate, max_attempts=3)
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.rewards), 2)
        self.assertEqual(len(result.sources), result.attempts)
        # attempt 2 pays the -0.5 retry penalty
        self.assertEqual(result.rewards, [0.0, 15.5])
        self.assertNotIn("this is not C", result.last_source)

    def test_max_attempts_exhausted(self):
        def generate(messages):
            return "```c\n" + BAD_C + "\n```"

        result = solve(ADD_TASK, generate, max_attempts=3)
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.rewards, [0.0, 0.0, 0.0])
        self.assertTrue(all(r["compiled"] is False for r in result.eval_results))

    def test_trajectory_logged_per_attempt(self):
        def generate(messages):
            if messages[-1]["role"] == "user" and "Attempt 1 failed" in messages[-1]["content"]:
                return "```c\n" + GOOD_C + "```"
            return "```c\n" + BAD_C + "\n```"

        with tempfile.TemporaryDirectory() as td:
            log_path = f"{td}/solver.jsonl"
            result = solve(ADD_TASK, generate, max_attempts=3, log_path=log_path)
            with open(log_path, encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
            self.assertEqual(len(lines), result.attempts)
            for i, line in enumerate(lines, start=1):
                entry = json.loads(line)
                self.assertEqual(entry["attempt"], i)
                self.assertEqual(entry["task_id"], "001")
                # the logged source is the extracted C code (fences removed)
                self.assertNotIn("```c", entry["source"])
            self.assertIn("this is not C", json.loads(lines[0])["source"])
            self.assertIn("a+b", json.loads(lines[1])["source"])

    def test_feedback_mentions_compiler_stderr(self):
        from arena.evaluate import evaluate

        fb = build_feedback(evaluate("int main(void){ bad }\n", ADD_TASK))
        self.assertIn("TCC compilation failed", fb)
        self.assertIn("stderr:", fb)


if __name__ == "__main__":
    unittest.main()