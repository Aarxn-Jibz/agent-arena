"""Bounded hostile programs; opt in with RUN_DOCKER_SANDBOX_TESTS=1."""

import os
import unittest

from arena.sandbox import SandboxConfig, run_c


@unittest.skipUnless(os.environ.get("RUN_DOCKER_SANDBOX_TESTS") == "1",
                     "requires local Docker sandbox image and daemon access")
class SandboxAdversarialTest(unittest.TestCase):
    def test_network_is_unavailable(self):
        source = '''#include <sys/socket.h>
#include <netinet/in.h>
int main(void) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {.sin_family=AF_INET, .sin_port=53};
    a.sin_addr.s_addr = htonl(0x01010101);
    return connect(s, (void *)&a, sizeof a) == 0;
}'''
        result = run_c(source)
        self.assertTrue(result.compiled)
        self.assertEqual(result.exit_code, 0)

    def test_cannot_write_outside_scratch(self):
        source = '''#include <stdio.h>
int main(void) {
    FILE *a=fopen("/scratch/ok", "w");
    FILE *b=fopen("/work/escape", "w");
    FILE *c=fopen("/etc/escape", "w");
    if(a) fclose(a);
    return !(a && !b && !c);
}'''
        result = run_c(source)
        self.assertTrue(result.compiled)
        self.assertEqual(result.exit_code, 0)

    def test_cannot_use_image_tools_or_read_host_files(self):
        source = '''#include <stdio.h>
#include <stdlib.h>
int main(void) {
    FILE *host=fopen("/home/jibin/5thsemhons/README.md", "r");
    int tool=system("/usr/bin/id > /scratch/tool-output");
    if(host) fclose(host);
    return host != 0 || tool == 0;
}'''
        result = run_c(source)
        self.assertTrue(result.compiled)
        self.assertEqual(result.exit_code, 0)

    def test_memory_limit(self):
        source = '''#include <stdlib.h>
int main(void) {
    volatile char *p=malloc(512*1024*1024);
    if(!p) return 0;
    for(unsigned long i=0;i<512UL*1024*1024;i+=4096) p[i]=1;
    return 1;
}'''
        result = run_c(source, config=SandboxConfig(memory_mb=64, timeout_seconds=2))
        self.assertTrue(result.compiled)
        self.assertNotEqual(result.exit_code, 1)
        self.assertFalse(result.timed_out)

    def test_pid_limit(self):
        source = '''#include <stdio.h>
#include <unistd.h>
int main(void) {
    int n=0;
    for(int i=0;i<80;i++) {
        int p=fork();
        if(p==0) {sleep(2); _exit(0);}
        if(p<0) break;
        n++;
    }
    printf("%d\\n",n);
    return 0;
}'''
        result = run_c(source, config=SandboxConfig(pids=16, timeout_seconds=3))
        self.assertTrue(result.compiled)
        self.assertEqual(result.exit_code, 0)
        self.assertLess(int(result.stdout.strip()), 80)

    def test_infinite_loop_is_stopped(self):
        result = run_c("int main(void){for(;;){} }",
                       config=SandboxConfig(timeout_seconds=1))
        self.assertTrue(result.compiled)
        self.assertTrue(result.timed_out)

    def test_output_is_bounded(self):
        source = '''#include <stdio.h>
int main(void) {for(;;) {puts("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"); fflush(stdout);} }'''
        result = run_c(source, config=SandboxConfig(output_bytes=4096, timeout_seconds=2))
        self.assertTrue(result.compiled)
        self.assertLessEqual(len(result.stdout.encode()), 4096)


if __name__ == "__main__":
    unittest.main()
