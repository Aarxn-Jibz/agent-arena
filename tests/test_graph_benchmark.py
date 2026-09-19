import os
import unittest

from arena.graph_benchmark import GraphBenchmark
from arena.sandbox import SandboxConfig

SINGLE_C = '''#include <stdio.h>
int main(void) {
    int n,m,d,w,q,u,v,x; char op[32];
    if(scanf("%d %d %d %d %d",&n,&m,&d,&w,&q)!=5) return 1;
    for(int i=0;i<m;i++) scanf("%d %d %d",&u,&v,&x);
    for(int i=0;i<q;i++) {scanf("%31s %d %d",op,&u,&v); puts("1");}
    return 0;
}'''


class GraphTest(unittest.TestCase):
    def setUp(self):
        self.b = GraphBenchmark()

    def test_oracle_and_generation(self):
        edges = {(0, 1): 2, (1, 2): 3}
        output = self.b.oracle(4, edges, [("reach", 0, 2), ("distance", 0, 2),
                                          ("distance", 0, 3), ("neighbors", 1, 0),
                                          ("components", 0, 0)], False)
        self.assertEqual(output, b"1\n5\n-1\n0,2\n2\n")
        challenge = self.b.initialize(4)
        self.assertEqual(self.b.generate_cases(challenge, 4), self.b.generate_cases(challenge, 4))
        self.assertFalse(self.b.validate_challenge({**challenge, "directed": True,
                                                    "operations": ["components"]})[0])

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_SANDBOX_TESTS") == "1", "Docker required")
    def test_docker_single_vertex(self):
        challenge = {**self.b.initialize(2), "vertices": 1, "density": 0,
                     "operations": ["reach"], "queries": 3}
        result = self.b.evaluate(SINGLE_C, challenge, SandboxConfig(output_bytes=4096))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["correctness"]["passed"], 1)
