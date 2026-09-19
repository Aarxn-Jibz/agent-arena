import os
import json
import tempfile
import unittest

from arena.http_benchmark import HttpBenchmark
from arena.sandbox import SandboxConfig

HEALTH_C = '''#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
int main(int argc,char **argv) {
    if(argc!=2) return 2;
    int s=socket(AF_UNIX,SOCK_STREAM,0);
    struct sockaddr_un a={0}; a.sun_family=AF_UNIX;
    strncpy(a.sun_path,argv[1],sizeof(a.sun_path)-1);
    if(bind(s,(void*)&a,sizeof(a)) || listen(s,8)) return 3;
    for(;;) {
        int c=accept(s,0,0); if(c<0) return 4;
        char buf[1024]; read(c,buf,sizeof(buf));
        const char *r="HTTP/1.1 200 OK\\r\\nContent-Length: 3\\r\\nContent-Type: text/plain\\r\\nConnection: close\\r\\n\\r\\nok\\n";
        write(c,r,strlen(r)); close(c);
    }
}'''


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.b = HttpBenchmark()

    def test_protocol_and_challenge(self):
        self.assertEqual(self.b.parse_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nContent-Type: text/plain\r\n\r\nok\n"),
            (200, b"ok\n", "text/plain"))
        self.assertIsNone(self.b.parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nok\n"))
        challenge = self.b.initialize(2)
        self.assertEqual(self.b.generate_requests(challenge), self.b.generate_requests(challenge))
        self.assertFalse(self.b.validate_challenge({**challenge, "concurrency": 9})[0])

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_SANDBOX_TESTS") == "1", "Docker required")
    def test_local_server_in_networkless_container(self):
        challenge = {**self.b.initialize(1), "routes": ["health"], "requests": 2}
        result = self.b.evaluate(HEALTH_C, challenge, SandboxConfig(output_bytes=16384))
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["correctness"]["passed"], 2)
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.b.record_episode(td, run_id="http-demo", episode_id=1,
                                            challenge=challenge, source=HEALTH_C,
                                            evaluation=result, git_before="before", git_after="after")
            self.assertEqual(json.loads(path.read_text())["benchmark"], "http")

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_SANDBOX_TESTS") == "1", "Docker required")
    def test_malformed_request_rejected_by_judge(self):
        challenge = {**self.b.initialize(1), "routes": ["health"], "requests": 1,
                     "malformed": True}
        result = self.b.evaluate(HEALTH_C, challenge, SandboxConfig(output_bytes=16384))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["correctness"]["passed"], 1)
