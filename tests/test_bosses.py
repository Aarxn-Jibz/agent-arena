import json
import os
import tempfile
import unittest

from arena.bosses import MiniShellBoss, TinyFilesystemBoss, freeze_zero_shot
from arena.sandbox import SandboxConfig

ECHO_C = '''#include <stdio.h>
#include <string.h>
int main(void){char line[128];if(!fgets(line,sizeof line,stdin))return 1;
 if(!strncmp(line,"echo ",5)){fputs(line+5,stdout);return 0;}return 2;}'''

EMPTY_FS_C = '''#include <stdio.h>
#include <string.h>
int main(int argc,char **argv){unsigned char image[4096];char cmd[64];
 if(argc!=2||strcmp(argv[1],"apply")||fread(image,1,4096,stdin)!=4096)return 1;
 if(!fgets(cmd,sizeof cmd,stdin))return 1;
 fwrite(image,1,4096,stdout);if(!strcmp(cmd,"LIST\\n")){putchar('\\n');return 0;}
 puts("ERR");return 0;}'''


class BossTest(unittest.TestCase):
    def test_shell_oracle_and_determinism(self):
        boss = MiniShellBoss()
        commands = ["pwd", "cd /work", "pwd", "run false", "status",
                    "set name value", "get name", "run echo child"]
        self.assertEqual(boss.oracle(commands), (b"/\n/work\n1\nvalue\nchild\n", 0))
        challenge = boss.initialize(4)
        self.assertEqual(boss.generate(challenge), boss.generate(challenge))
        self.assertFalse(boss.validate_challenge({**challenge, "commands": 0}))

    def test_virtual_image_oracle(self):
        boss = TinyFilesystemBoss()
        files = {}
        self.assertEqual(boss.oracle_step(files, "CREATE /a"), "OK\n")
        self.assertEqual(boss.oracle_step(files, "WRITE /a ff00"), "OK\n")
        self.assertEqual(boss.oracle_step(files, "READ /a"), "DATA ff00\n")
        self.assertEqual(boss.oracle_step(files, "LIST"), "/a\n")
        self.assertEqual(boss.oracle_step(files, "DELETE /a"), "OK\n")

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_SANDBOX_TESTS") == "1", "Docker required")
    def test_zero_shot_freeze_and_fresh_image_steps(self):
        shell = MiniShellBoss()
        seed = next(s for s in range(200) if shell.generate({"commands": 1, "seed": s,
                                                               "max_ms": 3000}) == ["echo hello"])
        challenge = {"commands": 1, "seed": seed, "max_ms": 3000}
        with tempfile.TemporaryDirectory() as td:
            path = freeze_zero_shot(td, shell, challenge, baseline_source=ECHO_C,
                                    trained_source=ECHO_C, baseline_commit="base",
                                    trained_commit="trained", config=SandboxConfig(output_bytes=8192))
            frozen = json.loads(path.read_text())
            self.assertTrue(frozen["baseline"]["passed"])
            self.assertTrue(frozen["trained"]["passed"])
            with self.assertRaises(FileExistsError):
                freeze_zero_shot(td, shell, challenge, baseline_source=ECHO_C,
                                 trained_source=ECHO_C, baseline_commit="base",
                                 trained_commit="trained")
        fs = TinyFilesystemBoss()
        seed = next(s for s in range(200) if fs.generate({"operations": 1, "seed": s,
                                                            "max_ms": 3000}) == ["LIST"])
        result = fs.evaluate(EMPTY_FS_C, {"operations": 1, "seed": seed, "max_ms": 3000},
                             SandboxConfig(output_bytes=8192))
        self.assertTrue(result["passed"], result)
