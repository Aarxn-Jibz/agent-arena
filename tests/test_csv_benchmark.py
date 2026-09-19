import json
import os
import tempfile
import unittest

from arena.csv_benchmark import CsvBenchmark, OPERATIONS
from arena.sandbox import SandboxConfig

COUNT_C = '''#include <stdio.h>
#include <string.h>
int main(int argc,char **argv) {
    if(argc!=4 || strcmp(argv[1],"count")) return 2;
    int c,lines=0;
    while((c=getchar())!=EOF) if(c=='\\n') lines++;
    printf("%d\\n",lines-1);
    return 0;
}'''


class CsvBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.b = CsvBenchmark()

    def test_every_operation_has_deterministic_oracle(self):
        for operation in OPERATIONS:
            challenge = {**self.b.initialize(7), "operation": operation,
                         "distribution": "quoted", "rows": 12, "column": 0}
            if operation == "uppercase":
                challenge["column"] = 1
            cases = self.b.generate_cases(challenge, 7)
            self.assertEqual(cases, self.b.generate_cases(challenge, 7))
            self.assertEqual(cases[0].expected, self.b.oracle(cases[0], challenge))
            self.assertGreater(len(cases[0].stdin), 0)

    def test_validation_and_malformed_case(self):
        challenge = {**self.b.initialize(1), "invalid": True}
        case = self.b.generate_cases(challenge, 1)[0]
        self.assertTrue(case.malformed)
        self.assertIsNone(case.expected)
        self.assertFalse(self.b.validate_challenge({**challenge, "column": 99})[0])
        self.assertFalse(self.b.validate_challenge({**challenge, "operation": "unknown"})[0])

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_SANDBOX_TESTS") == "1", "Docker required")
    def test_docker_count_and_evidence(self):
        challenge = {**self.b.initialize(3), "rows": 5}
        result = self.b.evaluate(COUNT_C, challenge, SandboxConfig(output_bytes=4096))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["correctness"]["passed"], 1)
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.b.record_episode(td, run_id="csv-demo", episode_id=1,
                                            challenge=challenge, source=COUNT_C,
                                            evaluation=result, git_before="before", git_after="after")
            saved = json.loads(path.read_text())
            self.assertEqual(saved["benchmark"], "csv")
            self.assertEqual(saved["performance"]["largest_workload_completed"], 5)
