"""Tests for the deterministic reward function (arena.evaluate.compute_reward)."""

import shutil
import unittest

from arena.evaluate import compute_reward, evaluate

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


class ComputeRewardTest(unittest.TestCase):
    def test_full_score_first_attempt(self):
        # 1 (compiled) + 10 (all tests) + 5 (all-pass bonus)
        self.assertEqual(compute_reward(3, 3, False, 0, attempt=1), 16.0)

    def test_attempt_penalty(self):
        self.assertEqual(compute_reward(3, 3, False, 0, attempt=2), 15.5)
        self.assertEqual(compute_reward(3, 3, False, 0, attempt=3), 15.0)

    def test_partial_score_is_proportional(self):
        # 1 + 10/3 = 4.3333
        self.assertEqual(compute_reward(1, 3, False, 0, attempt=1), 4.3333)

    def test_timeout_penalty(self):
        self.assertEqual(compute_reward(3, 3, True, 0, attempt=1), 6.0)

    def test_runtime_error_penalty_per_crash(self):
        self.assertEqual(compute_reward(0, 3, False, 3, attempt=1), -2.0)
        self.assertEqual(compute_reward(2, 3, False, 1, attempt=1), 6.6667)

    def test_deterministic_for_same_inputs(self):
        a = compute_reward(2, 5, False, 0, attempt=4)
        b = compute_reward(2, 5, False, 0, attempt=4)
        self.assertEqual(a, b)
        self.assertEqual(compute_reward(7, 7, False, 0, 1), 16.0)


@unittest.skipUnless(HAS_TCC, "tcc not on PATH")
class EvalRewardIntegrationTest(unittest.TestCase):
    def test_compile_failure_scores_zero(self):
        self.assertEqual(evaluate(SYNTAX_ERROR, ADD_TASK).reward, 0.0)

    def test_correct_code_scores_full(self):
        self.assertEqual(evaluate(ADD_CORRECT, ADD_TASK).reward, 16.0)

    def test_retry_penalty_reduces_reward(self):
        first = evaluate(ADD_CORRECT, ADD_TASK).reward
        retried = evaluate(ADD_CORRECT, ADD_TASK, attempt=2).reward
        self.assertEqual(first - retried, 0.5)

    def test_wrong_output_scores_less(self):
        self.assertLess(evaluate(ADD_WRONG, ADD_TASK).reward, 16.0)

    def test_timeout_penalty_applied(self):
        from dataclasses import asdict

        task = dict(ADD_TASK, timeout_seconds=1.0)
        result = evaluate(INFINITE_LOOP, task)
        self.assertTrue(result.timed_out)
        # 1 (compiled) + 0 tests - 5 (timeout)
        self.assertEqual(result.reward, -4.0)

    def test_evaluation_is_deterministic_end_to_end(self):
        from dataclasses import asdict

        def strip_timing(result: dict) -> dict:
            result = dict(result)
            result.pop("runtime_ms")
            result["tests"] = [{k: v for k, v in t.items() if k != "elapsed_ms"}
                               for t in result["tests"]]
            return result

        r1 = strip_timing(asdict(evaluate(ADD_CORRECT, ADD_TASK)))
        r2 = strip_timing(asdict(evaluate(ADD_CORRECT, ADD_TASK)))
        self.assertEqual(r1, r2)
        # timing itself must still be recorded on every run
        self.assertGreater(evaluate(ADD_CORRECT, ADD_TASK).runtime_ms, 0.0)


if __name__ == "__main__":
    unittest.main()