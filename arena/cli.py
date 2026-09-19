"""Tiny command-line interface for the arena.

Usage:
    python -m arena run <task.json> <solution.c> [--log PATH] [--attempt N] [--json]

Exit code is 0 when the solution passes every test, 1 otherwise (so retry
loops in scripts can stop on the first success).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .evaluate import evaluate, load_task
from .log import append_trajectory, trajectory_entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arena",
        description="Evaluate C source against a task using TCC, deterministically.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Evaluate a C file against a task.")
    run.add_argument("task", type=Path, help="Path to a task JSON file.")
    run.add_argument("source", type=Path, help="Path to the C source file.")
    run.add_argument(
        "--log", type=Path, default=None,
        help="Append one attempt record to this JSONL trajectory file.",
    )
    run.add_argument(
        "--attempt", type=int, default=1,
        help="1-based attempt number (applies the small retry penalty to reward).",
    )
    run.add_argument(
        "--json", action="store_true",
        help="Print the full evaluation result as JSON instead of a summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.attempt < 1:
        print("error: --attempt must be >= 1", file=sys.stderr)
        return 2
    task = load_task(args.task)
    source = Path(args.source).read_text(encoding="utf-8")
    result = evaluate(source, task, attempt=args.attempt)
    if args.log is not None:
        append_trajectory(args.log, trajectory_entry(task, args.attempt, source, asdict(result)))
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(summary(task, result))
    return 0 if result.success else 1


def summary(task: dict, result) -> str:
    if not result.compiled:
        status = "COMPILE ERROR"
    elif result.success:
        status = "PASS"
    else:
        status = "FAIL"
    lines = [f"task:    {task['id']} - {task['title']}", f"status:  {status}"]
    if not result.compiled:
        lines.append("compiler stderr:")
        lines.append(indent(result.compile_stderr))
        return "\n".join(lines)
    lines.append(f"tests:   {result.passed}/{result.total} passed")
    lines.append(f"runtime: {result.runtime_ms} ms")
    lines.append(f"reward:  {result.reward}")
    if result.timed_out:
        lines.append("note:    at least one test timed out")
    if not result.success:
        lines.append("first failing test:")
        for test in result.tests:
            if not test.passed:
                lines.append(indent(f"expected: {test.expected_output!r}"))
                lines.append(indent(f"got:      {test.stdout!r}"))
                if test.timed_out:
                    lines.append(indent("(timed out)"))
                elif test.exit_code != 0:
                    lines.append(indent(f"(exit code {test.exit_code})"))
                break
    return "\n".join(lines)


def indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.rstrip("\n").split("\n"))


if __name__ == "__main__":
    sys.exit(main())