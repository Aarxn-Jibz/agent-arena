"""Integration tests for TCC compilation and process execution.

These use the real tcc binary; they are skipped when tcc is not on PATH.
"""

import os
import shutil
import tempfile
import unittest

from arena.runner import compile_c, run_binary, tcc_path

HAS_TCC = shutil.which("tcc") is not None


@unittest.skipUnless(HAS_TCC, "tcc not on PATH")
class CompileRunTest(unittest.TestCase):
    def test_valid_source_compiles(self):
        with tempfile.TemporaryDirectory() as td:
            res = compile_c('#include <stdio.h>\nint main(void){puts("ok");return 0;}\n', td)
            self.assertTrue(res.success)
            self.assertEqual(res.exit_code, 0)

    def test_invalid_source_returns_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            res = compile_c("int main(void){ this is not C }\n", td)
            self.assertFalse(res.success)
            self.assertNotEqual(res.exit_code, 0)
            self.assertTrue(res.stderr.strip(), "expected compiler diagnostics")

    def test_compile_can_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            res = compile_c("int main(void){return 0;}\n", td, timeout=1e-9)
            self.assertFalse(res.success)
            self.assertIn("timed out", res.stderr)

    def test_run_captures_output_and_exit_code(self):
        with tempfile.TemporaryDirectory() as td:
            compile_c('#include <stdio.h>\nint main(void){int n;scanf("%d",&n);printf("%d\\n",n*2);return 3;}\n', td)
            r = run_binary(os.path.join(td, "solution"), "21\n", 5.0)
            self.assertFalse(r.timed_out)
            self.assertEqual(r.exit_code, 3)
            self.assertEqual(r.stdout, "42\n")

    def test_run_timeout_kills_infinite_loop(self):
        with tempfile.TemporaryDirectory() as td:
            compile_c("int main(void){ for(;;){} }\n", td)
            r = run_binary(os.path.join(td, "solution"), "", 0.5)
            self.assertTrue(r.timed_out)
            self.assertIsNone(r.exit_code)
            self.assertGreaterEqual(r.elapsed_ms, 400)

    def test_execution_timing_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            compile_c(
                '#include <stdio.h>\nint main(void){long long i;for(i=0;i<30000000;i++);puts("done");return 0;}\n',
                td,
            )
            r = run_binary(os.path.join(td, "solution"), "", 5.0)
            self.assertGreater(r.elapsed_ms, 0.0)
            self.assertEqual(r.stdout, "done\n")

    def test_tcc_not_found_raises_clear_error(self):
        old = os.environ.get("TCC_BIN")
        os.environ["TCC_BIN"] = "/nonexistent/tcc-binary"
        try:
            with self.assertRaisesRegex(RuntimeError, "tcc not found"):
                tcc_path()
        finally:
            if old is None:
                os.environ.pop("TCC_BIN", None)
            else:
                os.environ["TCC_BIN"] = old


if __name__ == "__main__":
    unittest.main()