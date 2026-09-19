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
    sources: list[str]
    log_path: str | None = None


_EXEMPLAR = (
    "Example of the exact required, minimal style (different problem - multiply two integers):\n"
    "```c\n"
    "#include <stdio.h>\n"
    'int main(void){int a,b;scanf("%d %d",&a,&b);printf("%d\\n",a*b);return 0;}\n'
    "```\n"
)


def build_prompt(task: dict) -> str:
    """System prompt for the Solver.

    The few-shot exemplar (labeled as a DIFFERENT problem) comes first to set
    the bare `printf("%d\\n", ...)` style; the real task comes last, right
    before the request for the code. Expected outputs of the real task's
    hidden tests are never shown.
    """
    return (
        "You are solving C programming problems. Write ONLY a complete compilable C\n"
        "program that reads from stdin and writes to stdout. The program must print\n"
        "EXACTLY the required output and nothing else (no prompts, no labels, no extra\n"
        "text). Do not explain.\n\n"
        + _EXEMPLAR
        + f"\nNow your task.\n\nProblem (id={task['id']}): {task['prompt']}\n\n"
        + "Write your solution for the problem above as exactly one C code block: ```c ... ```"
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
            lines.append(
                "Your program must print EXACTLY the expected output and nothing else."
            )
            lines.append(f"  input:    {test.input!r}")
            lines.append(f"  expected: {test.expected_output!r}")
            lines.append(f"  got:      {test.stdout!r}")
            if test.exit_code != 0:
                lines.append(f"  exit code: {test.exit_code}")
            if test.timed_out:
                lines.append("  (timed out)")
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
    sources: list[str] = []
    last_source: str | None = None
    for attempt in range(1, max_attempts + 1):
        reply = generate(messages)
        source = extract_c_code(reply) or reply
        last_source = source
        sources.append(source)
        result = evaluate(source, task, attempt=attempt)
        rewards.append(result.reward)
        eval_results.append(asdict(result))
        if log_path:
            append_trajectory(log_path, trajectory_entry(task, attempt, source, asdict(result)))
        if result.success:
            return SolverResult(True, attempt, last_source, rewards, eval_results, sources, log_path)
        feedback = f"Attempt {attempt} failed.\n{build_feedback(result)}\n"
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": feedback +
                         "\nFix the program and write the complete corrected C "
                         "code inside one ```c ... ``` block."})
    return SolverResult(False, max_attempts, last_source, rewards, eval_results, sources, log_path)
