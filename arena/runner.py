"""TCC compilation and process execution for the arena.

TCC is the only C compiler used here. The binary is resolved once via
`shutil.which` and can be overridden with the TCC_BIN environment variable.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import time

# Cap on how long a single compilation may take (pathological sources).
COMPILE_TIMEOUT_SECONDS = 30.0


class TCCNotFoundError(RuntimeError):
    """Raised when the tcc executable cannot be found on PATH."""


def tcc_path() -> str:
    """Resolve the tcc executable (override the name with TCC_BIN)."""
    name = os.environ.get("TCC_BIN", "tcc")
    resolved = shutil.which(name)
    if resolved is None:
        raise TCCNotFoundError(
            "tcc not found on PATH. Install it with `sudo apt-get install tcc`, "
            "or set TCC_BIN=/path/to/tcc."
        )
    return resolved


@dataclasses.dataclass
class CompileResult:
    exit_code: int
    success: bool
    stdout: str
    stderr: str


@dataclasses.dataclass
class RunResult:
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    elapsed_ms: float


def compile_c(source: str, workdir: str | os.PathLike, timeout: float = COMPILE_TIMEOUT_SECONDS) -> CompileResult:
    """Write `source` to solution.c inside `workdir` and compile it with tcc.

    The caller owns `workdir` (use a temporary directory for real runs).
    """
    workdir = os.fspath(workdir)
    src_path = os.path.join(workdir, "solution.c")
    exe_path = os.path.join(workdir, "solution")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(source)
    try:
        proc = subprocess.run(
            [tcc_path(), "solution.c", "-o", "solution"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(
            exit_code=-1,
            success=False,
            stdout="",
            stderr="tcc: compilation timed out",
        )
    return CompileResult(
        exit_code=proc.returncode,
        success=proc.returncode == 0 and os.path.exists(exe_path),
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def run_binary(exe_path: str | os.PathLike, stdin_text: str, timeout: float) -> RunResult:
    """Run a compiled binary once, feeding `stdin_text`, enforcing `timeout`.

    Timing uses time.perf_counter(); a TimeoutExpired kills the child and is
    reported as `timed_out=True` rather than an exception.
    """
    exe_path = os.fspath(exe_path)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [exe_path],
            cwd=os.path.dirname(exe_path),
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as err:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return RunResult(
            exit_code=None,
            timed_out=True,
            stdout=err.stdout or "",
            stderr=err.stderr or "",
            elapsed_ms=round(elapsed_ms, 3),
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return RunResult(
        exit_code=proc.returncode,
        timed_out=False,
        stdout=proc.stdout,
        stderr=proc.stderr,
        elapsed_ms=round(elapsed_ms, 3),
    )