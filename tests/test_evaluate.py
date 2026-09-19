"""Tests for task loading, output normalization and deterministic evaluation."""

import json
import os
import shutil
import tempfile
import unittest

from arena.evaluate import EvalResult, evaluate, load_task, normalize_output

HAS_TCC = shutil.which("tcc") is not None

ADD_TASK = {
    "id": "001",
    "title": "Add two integers",
    "prompt": "Read two integers and print their sum.",
    "timeout_seconds": 2,
    "tests": [
        {"input": "2 3\n", "expected_output": "5\n"},
        {"input": "-5 12\n", "expected_output": "7\n"},
        {"input": "0 0\n", "expected_output": "0\n"},
    ],
}

ADD_CORRECT = '#include <stdio.h>\nint main(void){int a,b;scanf("%d %d",&a,&b);printf("%d\\n",a+b);return 0;}\n'
ADD_WRONG = '#include <stdio.h>\nint main(void){int a,b;scanf("%d %d",&a,&b);printf("%d\\n",a-b);return 0;}\n'
SYNTAX_ERROR = "int main(void){ this is not C }\n"
INFINITE_LOOP = "int main(void){ for(;;){} return 0; }\n"


class NormalizeOutputTest(unittest.TestCase):
    def test_strips_trailing_whitespace_per_line(self):
        self.assertEqual(normalize_output("a  \nb\t\n"), "a\nb")
        self.assertEqual(normalize_output("a  \nb\t\n"), normalize_output("a\nb\n"))

    def test_normalizes_crlf(self):
        self.assertEqual(normalize_output("a\r\nb\r\n"), normalize_output("a\nb\n"))

    def test_drops_trailing_blank_lines_but_keeps_leading(self):
        # leading blank lines stay significant; trailing blank lines are noise
        self.assertEqual(normalize_output("\n\na\n\n"), "\n\na")

    def test_empty_output(self):
        self.assertEqual(normalize_output(""), "")


class LoadTaskTest(unittest.TestCase):
    def test_loads_inline_dict(self):
        self.assertEqual(load_task(ADD_TASK)["id"], "001")

    def test_loads_json_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(ADD_TASK, f)
            path = f.name
        try:
            self.assertEqual(load_task(path)["title"], "Add two integers")
        finally:
            os.unlink(path)

    def test_missing_field_raises(self):
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            load_task({"id": "x", "prompt": "p", "timeout_seconds": 1, "tests": [{"input": "", "expected_output": ""}]})

    def test_empty_tests_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty 'tests'"):
            load_task(dict(ADD_TASK, tests=[]))

    def test_bad_timeout_raises(self):
        with self.assertRaisesRegex(ValueError, "positive number"):
            load_task(dict(ADD_TASK, timeout_seconds=0))

    def test_test_without_expected_output_raises(self):
        bad = dict(ADD_TASK, tests=[{"input": "x"}])
        with self.assertRaisesRegex(ValueError, "input' and 'expected_output'"):
            load_task(bad)


@unittest.skipUnless(HAS_TCC, "tcc not on PATH")
class EvaluateTest(unittest.TestCase):
    def test_valid_code_passes_all_tests(self):
        result = evaluate(ADD_CORRECT, ADD_TASK)
        self.assertIsInstance(result, EvalResult)
        self.assertTrue(result.compiled)
        self.assertEqual(result.compile_exit_code, 0)
        self.assertEqual(result.passed, result.total)
        self.assertTrue(result.success)
        self.assertFalse(result.timed_out)

    def test_syntax_error_returns_compiler_diagnostics_without_tests(self):
        result = evaluate(SYNTAX_ERROR, ADD_TASK)
        self.assertFalse(result.compiled)
        self.assertNotEqual(result.compile_exit_code, 0)
        self.assertTrue(result.compile_stderr.strip())
        self.assertEqual(result.passed, 0)
        self.assertEqual(result.tests, [])
        self.assertFalse(result.success)

    def test_valid_but_wrong_output_fails_tests(self):
        result = evaluate(ADD_WRONG, ADD_TASK)
        self.assertTrue(result.compiled)
        self.assertLess(result.passed, result.total)
        self.assertFalse(result.success)

    def test_partial_score(self):
        task = dict(
            ADD_TASK,
            tests=[
                {"input": "2 3\n", "expected_output": "5\n"},
                {"input": "1 1\n", "expected_output": "99\n"},
            ],
        )
        result = evaluate(ADD_CORRECT, task)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.total, 2)
        self.assertFalse(result.success)

    def test_infinite_loop_times_out(self):
        task = dict(ADD_TASK, timeout_seconds=1.0)
        result = evaluate(INFINITE_LOOP, task)
        self.assertTrue(result.compiled)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.passed, 0)
        self.assertGreaterEqual(result.runtime_ms, 800)

    def test_runtime_error_fails_test_even_with_right_output(self):
        code = '#include <stdio.h>\nint main(void){printf("5\\n");return 1;}\n'
        result = evaluate(code, ADD_TASK)
        self.assertTrue(result.compiled)
        self.assertEqual(result.passed, 0)
        self.assertFalse(result.success)

    def test_execution_timing_recorded(self):
        code = '#include <stdio.h>\nint main(void){long long i;for(i=0;i<10000000;i++);return 0;}\n'
        result = evaluate(code, ADD_TASK)
        self.assertTrue(result.compiled)
        self.assertGreaterEqual(result.runtime_ms, 0.0)
        self.assertGreater(sum(t.elapsed_ms for t in result.tests), 0.0)


if __name__ == "__main__":
    unittest.main()