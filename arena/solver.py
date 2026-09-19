"""Solver: an LLM writes C code, the arena judges it, and the loop retries.

The retry loop (`solve`) is model-agnostic: it takes a `generate(messages)`
callable that maps a chat transcript (list of {"role", "content"} dicts) to a
model reply string. The transformers adapter lives in `solver_llm.py` and is
never imported by the arena judge, so LLM dependencies stay optional.

Feedback shown to the model never includes the expected test outputs: the
deterministic TCC/test environment is the only judge.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import asdict

from .evaluate import EvalResult, evaluate
from .log import append_trajectory, trajectory_entry


@dataclasses.dataclass
class SolverResult:
    success: bool
    attempts: int
    last_source: str | None
    rewards: list[float]
    eval_results: list[dict]
    log_path: str | None = None


def build_prompt(task: dict) -> str:
    """System + task statement. Expected test outputs are deliberately hidden."""
    return (
        "You are solving C programming problems. Write ONLY a complete compilable C\n"
        "program that reads from stdin and writes to stdout. Do not explain.\n\n"
        f"Problem (id={task['id']}): {task['prompt']}\n\n"
        "Return your answer as exactly one C code block: ```c ... ```"
    )


def build_feedback(result: EvalResult) -> str:
    """Compiler/test feedback for one attempt, shaped for the model."""
    if not result.compiled:
        return (
            f"TCC compilation failed (exit {result.compile_exit_code}).\n"
            f"compiler stderr:\n{result.compile_stderr}"
        )
    if result.success:
        return f"All {result.total} tests passed. reward={result.reward}."
    lines = [f"You passed {result.passed}/{result.total} tests. reward={result.reward}."]
    if result.timed_out:
        lines.append("At least one test timed out (program did not finish).")
    for test in result.tests:
        if not test.passed:
            lines.append(f"Failing test input:    {test.input!r}")
            lines.append(f"Expected stdout:       {test.expected_output!r}")
            lines.append(f"Your stdout:           {test.stdout!r}")
            lines.append(f"Your stderr:           {test.stderr!r}")
            lines.append(f"Exit code: {test.exit_code}, timed out: {test.timed_out}")
            break
    return "\n".join(lines)


def extract_c_code(reply: str) -> str | None:
    """Pull the C code out of a model reply (```c ... ``` fence)."""
    match = re.search(r"```(?:c|C)?\s*\n(.*?)```", reply, re.DOTALL)
    if not match:
        return None
    code = match.group(1).strip()
    if not code:
        return None
    return code + "\n"


def solve(task: dict, generate, max_attempts: int = 3,
          log_path: str | None = None) -> SolverResult:
    """Retry loop: generate -> arena evaluate -> feedback -> retry.

    `generate(messages)` receives the transcript so far (list of
    {"role", "content"} dicts) and returns the model's reply text.
    Stops on success or when `max_attempts` are exhausted.
    """
    messages = [{"role": "user", "content": build_prompt(task)}]
    rewards: list[float] = []
    eval_results: list[dict] = []
    last_source: str | None = None
    for attempt in range(1, max_attempts + 1):
        reply = generate(messages)
        source = extract_c_code(reply) or reply
        last_source = source
        result = evaluate(source, task, attempt=attempt)
        rewards.append(result.reward)
        eval_results.append(asdict(result))
        if log_path:
            append_trajectory(log_path, trajectory_entry(task, attempt, source, asdict(result)))
        if result.success:
            return SolverResult(True, attempt, last_source, rewards, eval_results, log_path)
        feedback = f"Attempt {attempt} failed.\n{build_feedback(result)}\n"
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": feedback +
                         "\nFix the program and write the complete corrected C "
                         "code inside one ```c ... ``` block."})
    return SolverResult(False, max_attempts, last_source, rewards, eval_results, log_path)