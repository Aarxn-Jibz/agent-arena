"""Expression-language benchmark with a restricted, exact integer oracle."""

from __future__ import annotations

import ast
import random
import statistics
from datetime import datetime, timezone

from .bench_result import record_judgement
from .benchmark import Case
from .sandbox import SandboxConfig, run_c

OPS = ("+", "-", "*", "/")
MIN_INT, MAX_INT = -(2**63), 2**63 - 1


def _now():
    return datetime.now(timezone.utc).isoformat()


def _checked(value):
    if not MIN_INT <= value <= MAX_INT:
        raise ValueError("integer overflow")
    return value


def oracle(expression: str) -> bytes:
    """Only the public language is evaluated; all other Python AST nodes fail."""
    try:
        tree = ast.parse(expression, mode="eval")

        def walk(node):
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return _checked(node.value)
            if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
                value = walk(node.operand)
                return _checked(value if isinstance(node.op, ast.UAdd) else -value)
            if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
                left, right = walk(node.left), walk(node.right)
                if isinstance(node.op, ast.Add):
                    return _checked(left + right)
                if isinstance(node.op, ast.Sub):
                    return _checked(left - right)
                if isinstance(node.op, ast.Mult):
                    return _checked(left * right)
                if right == 0:
                    raise ValueError("division by zero")
                quotient = abs(left) // abs(right)
                return _checked(-quotient if (left < 0) != (right < 0) else quotient)
            raise ValueError("unsupported syntax")

        return f"{walk(tree.body)}\n".encode()
    except (SyntaxError, ValueError, RecursionError):
        return b"ERROR\n"


class ExpressionBenchmark:
    name = "expression"

    def initialize(self, seed: int) -> dict:
        return {"operators": list(OPS), "allow_unary": True, "max_depth": 3,
                "max_tokens": 12, "number_bound": 100, "cases": 8,
                "invalid_cases": 2, "max_ms": 3000, "seed": seed}

    def validate_challenge(self, challenge: dict) -> tuple[bool, str]:
        if not isinstance(challenge, dict) or set(challenge) != set(self.initialize(0)):
            return False, "invalid challenge fields"
        ops = challenge["operators"]
        if not isinstance(ops, list) or not ops or len(ops) > 4 or len(set(map(str, ops))) != len(ops) or any(op not in OPS for op in ops):
            return False, "invalid operators"
        if not isinstance(challenge["allow_unary"], bool):
            return False, "allow_unary must be boolean"
        for key, low, high in (("max_depth", 1, 8), ("max_tokens", 1, 60),
                               ("number_bound", 0, 1000000), ("cases", 1, 20),
                               ("invalid_cases", 0, 20), ("max_ms", 1, 10000),
                               ("seed", 0, 2**32-1)):
            value = challenge[key]
            if type(value) is not int or not low <= value <= high:
                return False, f"{key} out of range"
        if challenge["invalid_cases"] > challenge["cases"]:
            return False, "invalid_cases exceeds cases"
        return True, "valid"

    def generate_cases(self, challenge: dict, seed: int) -> list[Case]:
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        rng = random.Random(seed)

        def build(depth, budget):
            if depth == 0 or budget < 3 or rng.randrange(3) == 0:
                value = str(rng.randrange(challenge["number_bound"] + 1))
                if challenge["allow_unary"] and rng.randrange(5) == 0:
                    value = rng.choice(("+", "-")) + value
                return value
            left_budget = rng.randrange(1, budget - 1)
            left = build(depth - 1, left_budget)
            right = build(depth - 1, budget - 1 - left_budget)
            return f"({left} {rng.choice(challenge['operators'])} {right})"

        malformed = ("1 +", "(2 * 3", "7 ** 2", "abc", "1 2", "()", "4 / / 2")
        cases = []
        for index in range(challenge["cases"]):
            invalid = index < challenge["invalid_cases"]
            expression = malformed[rng.randrange(len(malformed))] if invalid else build(
                challenge["max_depth"], challenge["max_tokens"])
            cases.append(Case(f"expression-{index}", (expression + "\n").encode(),
                              oracle(expression), invalid))
        return cases

    def correctness(self, case: Case, result) -> dict:
        return {"case": case.name, "passed": result.compiled and result.exit_code == 0 and
                not result.timed_out and result.stdout_bytes == case.expected,
                "malformed": case.malformed}

    def performance(self, case: Case, result) -> dict:
        return {"case": case.name, "elapsed_ms": result.elapsed_ms,
                "exit_code": result.exit_code, "timed_out": result.timed_out}

    def reward_inputs(self, correctness: list[dict], performance: list[dict]) -> dict:
        passed, total = sum(x["passed"] for x in correctness), len(correctness)
        eligible = total > 0 and passed == total
        return {"correctness_passed": passed, "correctness_total": total,
                "eligible_for_performance": eligible,
                "solver_reward": 1 + 1 / (1 + statistics.median(x["elapsed_ms"] for x in performance))
                if eligible else passed / max(1, total),
                "challenger_reward": 1 - passed / max(1, total)}

    def summarize(self, correctness: list[dict], performance: list[dict]) -> str:
        return f"{sum(x['passed'] for x in correctness)}/{len(correctness)} expressions passed"

    def evaluate(self, source: str, challenge: dict, config: SandboxConfig = SandboxConfig()) -> dict:
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        started_at = _now()
        checks, metrics, build = [], [], None
        cases = self.generate_cases(challenge, challenge["seed"])
        for case in cases:
            result = run_c(source, case.stdin, config)
            if build is None:
                build = {"exit_code": result.compile_exit_code, "stdout": result.compile_stdout,
                         "stderr": result.compile_stderr}
            checks.append(self.correctness(case, result))
            metrics.append({**self.performance(case, result), "input_bytes": len(case.stdin),
                            "within_time_limit": result.elapsed_ms <= challenge["max_ms"]})
        rewards = self.reward_inputs(checks, metrics)
        accepted = rewards["eligible_for_performance"] and all(x["within_time_limit"] for x in metrics)
        return {"started_at": started_at, "finished_at": _now(), "build": build,
                "correctness": {"passed": rewards["correctness_passed"], "total": len(checks),
                                "cases": checks},
                "performance": {"cases": metrics,
                                "median_ms": statistics.median(x["elapsed_ms"] for x in metrics),
                                "maximum_validated_input_bytes": max((len(case.stdin) for case, check in zip(cases, checks)
                                                                       if check["passed"]), default=0),
                                "memory_bytes": None},
                "reward_inputs": rewards, "feedback": self.summarize(checks, metrics),
                "accepted": accepted}

    def record_episode(self, root, *, run_id: str, episode_id: int, challenge: dict,
                       source: str, evaluation: dict, git_before: str, git_after: str | None,
                       solver_response: str = "", rationale: str | None = None,
                       patch: str | None = None):
        return record_judgement(root, benchmark=self.name, run_id=run_id,
                                episode_id=episode_id, challenge=challenge, source=source,
                                evaluation=evaluation, git_before=git_before, git_after=git_after,
                                solver_response=solver_response, rationale=rationale, patch=patch)
