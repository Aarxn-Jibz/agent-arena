import json
import os
import tempfile
import unittest

from arena.compression import CompressionBenchmark, KINDS
from arena.sandbox import SandboxConfig

COPY_C = '''#include <stdio.h>
#include <string.h>
int main(int argc,char **argv) {
    unsigned char b[4096]; size_t n;
    if(argc!=2 || (strcmp(argv[1],"compress") && strcmp(argv[1],"decompress"))) return 2;
    while((n=fread(b,1,sizeof b,stdin))>0) if(fwrite(b,1,n,stdout)!=n) return 3;
    return ferror(stdin) ? 4 : 0;
}'''


class CompressionTest(unittest.TestCase):
    def setUp(self):
        self.benchmark = CompressionBenchmark()

    def test_validation_and_deterministic_corpora(self):
        b = self.benchmark
        challenge = b.initialize(9)
        self.assertTrue(b.validate_challenge(challenge)[0])
        cases = b.generate_cases(challenge, 9)
        self.assertEqual([c.name for c in cases], [f"{k}-0" for k in KINDS])
        self.assertEqual(cases, b.generate_cases(challenge, 9))
        self.assertTrue(any(b"\x00" in c.stdin for c in cases))
        self.assertEqual(b._generate("edge", 0, 9), b"")
        self.assertEqual(len(b._generate("edge", 1, 9)), 1)
        self.assertEqual(len(b._generate("edge", 8192, 9)), 8192)
        for invalid in ({**challenge, "size": 8193}, {**challenge, "repeats": 0},
                        {**challenge, "distributions": ["unknown"]}):
            self.assertFalse(b.validate_challenge(invalid)[0])

    def test_correctness_first_reward(self):
        b = self.benchmark
        checks = [{"passed": True}, {"passed": False}]
        metrics = [{"original_bytes": 100, "bytes_saved": 80}]
        result = b.reward_inputs(checks, metrics)
        self.assertFalse(result["eligible_for_performance"])
        self.assertEqual(result["solver_reward"], 0.5)

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_SANDBOX_TESTS") == "1", "Docker required")
    def test_binary_roundtrip_and_episode_evidence(self):
        b = self.benchmark
        challenge = {"distributions": ["binary", "edge", "entropy"], "size": 64,
                     "seed": 5, "repeats": 1}
        config = SandboxConfig(output_bytes=4096)
        evaluated = b.evaluate(COPY_C, challenge, config)
        self.assertTrue(evaluated["accepted"])
        self.assertEqual(evaluated["correctness"]["passed"], 3)
        self.assertTrue(all(x["bytes_saved"] == 0 for x in evaluated["performance"]["cases"]))
        with tempfile.TemporaryDirectory() as td:
            json_path, md_path = b.record_episode(td, run_id="compression-demo", episode_id=1,
                                                   challenge=challenge, source=COPY_C,
                                                   git_before="before", git_after="after",
                                                   solver_response="copy bytes", evaluated=evaluated)
            record = json.loads(json_path.read_text())
            self.assertEqual(record["benchmark"], "compression")
            self.assertEqual(record["correctness"]["passed"], 3)
            self.assertEqual(record["input_generation"]["config"], challenge)
            self.assertTrue(md_path.exists())
            self.assertIn("### Performance", md_path.read_text())
            self.assertEqual(len((json_path.parent / "events.jsonl").read_text().splitlines()), 3)

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_SANDBOX_TESTS") == "1", "Docker required")
    def test_changed_bytes_are_rejected(self):
        challenge = {"distributions": ["binary"], "size": 16, "seed": 3, "repeats": 1}
        source = 'int main(void){return 0;}'
        result = self.benchmark.evaluate(source, challenge, SandboxConfig(output_bytes=4096))
        self.assertFalse(result["accepted"])
        self.assertFalse(result["reward_inputs"]["eligible_for_performance"])
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.benchmark.record_episode(td, run_id="reject", episode_id=1,
                                                    challenge=challenge, source=source,
                                                    git_before="before", git_after=None,
                                                    evaluated=result)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["outcome"], "rejected")
            self.assertIsNone(saved["git_after"])
