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


# Reward shaping constants. Deterministic and intentionally simple so the
# scoring is easy to explain: every input maps to exactly one reward value.
REWARD_FULL_TEST_SCORE = 10.0  # proportional: 10 * passed / total
COMPILE_REWARD = 1.0           # small reward for producing compiling code
ALL_PASS_BONUS = 5.0           # bonus when every test passes without timeout
TIMEOUT_PENALTY = -5.0         # any timeout forfeits points
RUNTIME_ERROR_PENALTY = -1.0   # per test that exits with a non-zero code
ATTEMPT_PENALTY = -0.5         # small cost for each retry beyond the first


def compute_reward(passed: int, total: int, timed_out: bool,
                   runtime_errors: int, attempt: int = 1) -> float:
    """Deterministic reward for one evaluated attempt.

    reward = 1 (compiled) + 10 * passed/total
             + 5 (all pass, no timeout)
             - 5 (any timeout) - 1 * runtime_errors
             - 0.5 * (attempt - 1)

    Compile failures get 0.0 and never reach this function.
    """
    score = COMPILE_REWARD + REWARD_FULL_TEST_SCORE * passed / total
    if total > 0 and passed == total and not timed_out:
        score += ALL_PASS_BONUS
    if timed_out:
        score += TIMEOUT_PENALTY
    score += RUNTIME_ERROR_PENALTY * runtime_errors
    score += ATTEMPT_PENALTY * (attempt - 1)
    return round(score, 4)


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
    reward: float
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
                reward=0.0,
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
        timed_out = any(r.timed_out for r in results)
        runtime_errors = sum(1 for r in results if not r.timed_out and r.exit_code != 0)
        return EvalResult(
            compiled=True,
            compile_exit_code=comp.exit_code,
            compile_stdout=comp.stdout,
            compile_stderr=comp.stderr,
            passed=passed,
            total=len(results),
            runtime_ms=round(sum(r.elapsed_ms for r in results), 3),
            timed_out=timed_out,
            program_exit_code=results[-1].exit_code,
            program_stdout=results[-1].stdout,
            program_stderr=results[-1].stderr,
            tests=results,
            reward=compute_reward(passed, len(results), timed_out, runtime_errors, attempt),
            success=passed == len(results),
        )