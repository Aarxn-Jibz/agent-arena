"""Tiny command-line interface for the arena.

Usage:
    python -m arena run <task.json> <solution.c> [--log PATH] [--attempt N] [--json]
    python -m arena solve <task.json> [--attempts N] [--log PATH] [--model ID]
    python -m arena marl [--episodes N] [--attempts N] [--alpha A] [--gamma G]
                         [--epsilon E] [--seed S] [--state PATH] [--log PATH]

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
    sub.add_parser("experiment", help="Run the offline Docker-only autonomous experiment.")
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
    solve = sub.add_parser("solve", help="Run the LLM Solver retry loop on a task.")
    solve.add_argument("task", type=Path, help="Path to a task JSON file.")
    solve.add_argument(
        "--attempts", type=int, default=3,
        help="Maximum number of generation attempts (default: 3).",
    )
    solve.add_argument(
        "--log", type=Path, default=None,
        help="Append every attempt to this JSONL trajectory file.",
    )
    solve.add_argument(
        "--model", default=None,
        help="HuggingFace model id (default: HuggingFaceTB/SmolLM2-360M-Instruct).",
    )
    marl = sub.add_parser(
        "marl",
        help="Run adversarial MARL episodes (Challenger + Solver Q-learning).",
    )
    marl.add_argument(
        "--episodes", type=int, default=9,
        help="Number of MARL episodes (default: 9).",
    )
    marl.add_argument(
        "--attempts", type=int, default=2,
        help="Maximum Solver attempts per episode (default: 2).",
    )
    marl.add_argument(
        "--alpha", type=float, default=0.4,
        help="Q-learning learning rate (default: 0.4).",
    )
    marl.add_argument(
        "--gamma", type=float, default=0.8,
        help="Q-learning discount factor (default: 0.8).",
    )
    marl.add_argument(
        "--epsilon", type=float, default=0.3,
        help="Epsilon-greedy exploration rate (default: 0.3).",
    )
    marl.add_argument(
        "--seed", type=int, default=42,
        help="Seeded RNG for epsilon-greedy policies (default: 42).",
    )
    marl.add_argument(
        "--state", type=Path, default=Path("marl_state.json"),
        help="Q-table/state persistence file (default: marl_state.json).",
    )
    marl.add_argument(
        "--log", type=Path, default=Path("trajectories/marl.jsonl"),
        help="Episode JSONL log (default: trajectories/marl.jsonl).",
    )
    marl.add_argument(
        "--report", type=Path, default=Path("trajectories/marl-report.md"),
        help="Human-readable run report (default: trajectories/marl-report.md).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    actual = sys.argv[1:] if argv is None else argv
    if actual and actual[0] == "experiment":
        from .experiment import main as experiment_main
        experiment_main(actual[1:])
        return 0
    args = build_parser().parse_args(argv)
    if args.command == "solve":
        return _cmd_solve(args)
    if args.command == "marl":
        return _cmd_marl(args)
    if args.attempt < 1:
        print("error: --attempt must be >= 1", file=sys.stderr)
        return 2
    try:
        task = load_task(args.task)
        source = Path(args.source).read_text(encoding="utf-8")
        result = evaluate(source, task, attempt=args.attempt)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.log is not None:
        append_trajectory(args.log, trajectory_entry(task, args.attempt, source, asdict(result)))
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(summary(task, result))
    return 0 if result.success else 1


def _cmd_solve(args) -> int:
    task = load_task(args.task)
    from .solver import solve
    from .solver_llm import llm_generate, load_model

    if args.model is None:
        model, tokenizer = load_model()
    else:
        model, tokenizer = load_model(args.model)
    result = solve(
        task,
        lambda messages: llm_generate(model, tokenizer, messages),
        max_attempts=args.attempts,
        log_path=args.log,
    )
    for attempt, (source, eval_res, reward) in enumerate(
        zip(result.sources, result.eval_results, result.rewards), start=1
    ):
        print(f"=== attempt {attempt} ===")
        print("generated C:")
        print(source)
        print(f"compiled:  {'yes' if eval_res['compiled'] else 'no'}")
        if eval_res["compile_stderr"]:
            print("compiler stderr:")
            print(eval_res["compile_stderr"])
        print(f"tests:     {eval_res['passed']}/{eval_res['total']} passed")
        print(f"runtime:   {eval_res['runtime_ms']} ms")
        print(f"reward:    {reward}")
        print()
    print(f"task:     {task['id']} - {task['title']}")
    print(f"success:  {'yes' if result.success else 'no'}  after {result.attempts} attempt(s)")
    print(f"rewards:  {result.rewards}")
    if result.log_path:
        print(f"log:      {result.log_path}")
    return 0 if result.success else 1


def _cmd_marl(args) -> int:
    from .marl import STATES, STRATEGIES, TASK_IDS, format_report, run_episodes, summarize_run
    from .solver_llm import llm_generate, load_model

    # The Challenger action space is the existing sample-task set.
    task_paths = sorted(Path("tasks").glob("*.json"))
    tasks = {load_task(p)["id"]: load_task(p) for p in task_paths}
    if not tasks:
        print("error: no tasks found under tasks/", file=sys.stderr)
        return 2

    # Load the frozen model ONCE, before the episode loop.
    print("loading model (once)...")
    model, tokenizer = load_model()

    def generate(messages):
        return llm_generate(model, tokenizer, messages)

    def print_episode(record):
        passed, total = record["best_tests"]
        print(f"Episode {record['episode']} | task {record['challenger_action']} | "
              f"prompt {record['solver_action']} | tests {passed}/{total} | "
              f"best reward {max(record['arena_rewards']):.2f} | "
              f"Challenger {record['challenger_reward']:.3f} | "
              f"Solver {record['solver_reward']:.3f} | state {record['next_state']}")

    records, state = run_episodes(
        tasks, generate,
        episodes=args.episodes,
        attempts=args.attempts,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
        state_path=args.state,
        log_path=args.log,
        on_episode=print_episode,
    )

    summary = summarize_run(records, state)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(format_report(summary, state, args.attempts, args.state,
                                         args.log, args.report),
                           encoding="utf-8")
    print("\nMARL RUN COMPLETE")
    print(f"Episodes: {summary['episodes']} ({summary['start_episode']}–{summary['end_episode']})")
    print(f"Tasks attempted: {sum(summary['tasks'].values())}")
    print(f"Average tests passed: {summary['average_tests_passed']:.1%}")
    print(f"Average Solver reward: {summary['average_solver_reward']:.3f}")
    print(f"Average Challenger reward: {summary['average_challenger_reward']:.3f}")
    print("Task selection:")
    for action in TASK_IDS:
        print(f"  {action}: {summary['tasks'][action]}")
    print("Prompt selection:")
    for action in STRATEGIES:
        print(f"  {action}: {summary['prompts'][action]}")
    print("Final Q-policy:")
    for s in STATES:
        print(f"  state {s}: task {summary['policy'][s]['task']}, "
              f"prompt {summary['policy'][s]['prompt']}")
    print(f"Report: {args.report}")
    return 0


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
