import os
import unittest

from arena.expression import ExpressionBenchmark, oracle
from arena.sandbox import SandboxConfig

LITERAL_C = '''#include <stdio.h>
int main(void){long long x; if(scanf("%lld",&x)!=1) return 1; printf("%lld\\n",x); return 0;}'''


class ExpressionTest(unittest.TestCase):
    def setUp(self):
        self.b = ExpressionBenchmark()

    def test_exact_oracle(self):
        self.assertEqual(oracle("1 + 2 * 3"), b"7\n")
        self.assertEqual(oracle("(1 + 2) * 3"), b"9\n")
        self.assertEqual(oracle("-7 / 2"), b"-3\n")
        self.assertEqual(oracle("1 / 0"), b"ERROR\n")
        self.assertEqual(oracle("9223372036854775807 + 1"), b"ERROR\n")
        self.assertEqual(oracle("1 ** 2"), b"ERROR\n")

    def test_challenges_and_cases(self):
        challenge = self.b.initialize(7)
        self.assertTrue(self.b.validate_challenge(challenge)[0])
        self.assertEqual(self.b.generate_cases(challenge, 7), self.b.generate_cases(challenge, 7))
        self.assertEqual(len(self.b.generate_cases(challenge, 7)), 8)
        self.assertFalse(self.b.validate_challenge({**challenge, "max_depth": 9})[0])
        self.assertFalse(self.b.validate_challenge({**challenge, "operators": ["**"]})[0])

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_SANDBOX_TESTS") == "1", "Docker required")
    def test_docker_literal_case(self):
        challenge = {**self.b.initialize(3), "operators": ["+"], "allow_unary": False,
                     "max_depth": 1, "max_tokens": 1, "cases": 1, "invalid_cases": 0}
        result = self.b.evaluate(LITERAL_C, challenge, SandboxConfig(output_bytes=4096))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["correctness"]["passed"], 1)
