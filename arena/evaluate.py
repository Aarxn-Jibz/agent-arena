"""Task loading and deterministic test evaluation for the arena."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path

from .runner import compile_c, run_binary

REQUIRED_TASK_FIELDS = ("id", "title", "prompt", "timeout_seconds", "tests")


def load_task(source: str | os.PathLike | dict) -> dict:
    """Load and validate a task from a JSON file path or an inline dict."""
    if isinstance(source, dict):
        task = source
    else:
        with open(Path(source), encoding="utf-8") as f:
            task = json.load(f)
    missing = [k for k in REQUIRED_TASK_FIELDS if k not in task]
    if missing:
        raise ValueError(f"task missing required fields: {', '.join(missing)}")
    if not isinstance(task["tests"], list) or not task["tests"]:
        raise ValueError("task must contain a non-empty 'tests' list")
    timeout = task["timeout_seconds"]
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("task 'timeout_seconds' must be a positive number")
    for test in task["tests"]:
        if not isinstance(test, dict) or "input" not in test or "expected_output" not in test:
            raise ValueError(f"test case in task {task['id']} needs 'input' and 'expected_output'")
    return task


def normalize_output(text: str) -> str:
    """Normalize output for comparison: CRLF -> LF, strip trailing whitespace
    per line, and drop trailing blank lines."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


@dataclasses.dataclass
class TestResult:
    input: str
    expected_output: str
    passed: bool
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    elapsed_ms: float


@dataclasses.dataclass
class EvalResult:
    compiled: bool
    compile_exit_code: int | None
    compile_stdout: str
    compile_stderr: str
    passed: int
    total: int
    runtime_ms: float
    timed_out: bool
    program_exit_code: int | None
    program_stdout: str
    program_stderr: str
    tests: list[TestResult]
    success: bool


def evaluate(source: str, task: dict, attempt: int = 1) -> EvalResult:
    """Compile `source` with TCC once, run every test case, and score it.

    A test passes only if the program exits 0, does not time out, and its
    normalized stdout equals the normalized expected output. Compilation runs
    in a private temporary directory that is cleaned up automatically.
    """
    with tempfile.TemporaryDirectory(prefix="arena-") as workdir:
        comp = compile_c(source, workdir)
        if not comp.success:
            return EvalResult(
                compiled=False,
                compile_exit_code=comp.exit_code,
                compile_stdout=comp.stdout,
                compile_stderr=comp.stderr,
                passed=0,
                total=len(task["tests"]),
                runtime_ms=0.0,
                timed_out=False,
                program_exit_code=None,
                program_stdout="",
                program_stderr="",
                tests=[],
                success=False,
            )
        results = []
        for test in task["tests"]:
            run = run_binary(
                os.path.join(workdir, "solution"), test["input"], task["timeout_seconds"]
            )
            passed = (
                not run.timed_out
                and run.exit_code == 0
                and normalize_output(run.stdout) == normalize_output(test["expected_output"])
            )
            results.append(
                TestResult(
                    input=test["input"],
                    expected_output=test["expected_output"],
                    passed=passed,
                    exit_code=run.exit_code,
                    timed_out=run.timed_out,
                    stdout=run.stdout,
                    stderr=run.stderr,
                    elapsed_ms=run.elapsed_ms,
                )
            )
        passed = sum(r.passed for r in results)
        return EvalResult(
            compiled=True,
            compile_exit_code=comp.exit_code,
            compile_stdout=comp.stdout,
            compile_stderr=comp.stderr,
            passed=passed,
            total=len(results),
            runtime_ms=round(sum(r.elapsed_ms for r in results), 3),
            timed_out=any(r.timed_out for r in results),
            program_exit_code=results[-1].exit_code,
            program_stdout=results[-1].stdout,
            program_stderr=results[-1].stderr,
            tests=results,
            success=passed == len(results),
        )