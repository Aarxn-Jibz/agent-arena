"""Deterministic graph CLI benchmark and trusted Python oracle."""

from __future__ import annotations

import heapq
import random
import statistics
from datetime import datetime, timezone

from .bench_result import record_judgement
from .benchmark import Case
from .sandbox import SandboxConfig, run_c

OPERATIONS = ("reach", "distance", "neighbors", "components")
TOPOLOGIES = ("random", "path", "cycle", "star", "disconnected")


def _now():
    return datetime.now(timezone.utc).isoformat()


class GraphBenchmark:
    name = "graph"

    def initialize(self, seed: int) -> dict:
        return {"vertices": 12, "density": 15, "topology": "random",
                "directed": False, "weighted": False, "operations": ["reach", "distance"],
                "queries": 12, "max_ms": 3000, "seed": seed}

    def validate_challenge(self, challenge: dict) -> tuple[bool, str]:
        if not isinstance(challenge, dict) or set(challenge) != set(self.initialize(0)):
            return False, "invalid challenge fields"
        if challenge["topology"] not in TOPOLOGIES:
            return False, "unsupported topology"
        if type(challenge["directed"]) is not bool or type(challenge["weighted"]) is not bool:
            return False, "directed and weighted must be booleans"
        ops = challenge["operations"]
        if not isinstance(ops, list) or not ops or len(ops) > 4 or len(set(map(str, ops))) != len(ops) or any(x not in OPERATIONS for x in ops):
            return False, "invalid operations"
        if challenge["directed"] and "components" in ops:
            return False, "components requires undirected graph"
        for key, low, high in (("vertices", 1, 200), ("density", 0, 100),
                               ("queries", 1, 100), ("max_ms", 1, 10000),
                               ("seed", 0, 2**32-1)):
            value = challenge[key]
            if type(value) is not int or not low <= value <= high:
                return False, f"{key} out of range"
        return True, "valid"

    def generate_cases(self, challenge: dict, seed: int) -> list[Case]:
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        rng = random.Random(seed)
        n = challenge["vertices"]
        edges = {}

        def add(u, v):
            if u == v:
                return
            key = (u, v) if challenge["directed"] else tuple(sorted((u, v)))
            edges[key] = rng.randrange(1, 21) if challenge["weighted"] else 1

        topology = challenge["topology"]
        if topology in ("path", "cycle"):
            for u in range(n - 1):
                add(u, u + 1)
            if topology == "cycle" and n > 2:
                add(n - 1, 0)
        elif topology == "star":
            for v in range(1, n):
                add(0, v)
        elif topology == "disconnected":
            for u in range(n - 1):
                if u != n // 2 - 1:
                    add(u, u + 1)
        for u in range(n):
            for v in range(n):
                if u == v or (not challenge["directed"] and v < u):
                    continue
                if topology == "disconnected" and (u < n // 2) != (v < n // 2):
                    continue
                if rng.randrange(100) < challenge["density"]:
                    add(u, v)
        queries = [(rng.choice(challenge["operations"]), rng.randrange(n), rng.randrange(n))
                   for _ in range(challenge["queries"])]
        lines = [f"{n} {len(edges)} {int(challenge['directed'])} {int(challenge['weighted'])} {len(queries)}"]
        lines += [f"{u} {v} {w}" for (u, v), w in sorted(edges.items())]
        lines += [f"{op} {u} {v}" for op, u, v in queries]
        return [Case("graph-0", ("\n".join(lines) + "\n").encode(),
                     self.oracle(n, edges, queries, challenge["directed"]))]

    @staticmethod
    def oracle(n, edges, queries, directed) -> bytes:
        adjacent = [[] for _ in range(n)]
        for (u, v), weight in edges.items():
            adjacent[u].append((v, weight))
            if not directed:
                adjacent[v].append((u, weight))

        def distances(start):
            distance = [None] * n
            distance[start] = 0
            queue = [(0, start)]
            while queue:
                current, u = heapq.heappop(queue)
                if current != distance[u]:
                    continue
                for v, weight in adjacent[u]:
                    next_distance = current + weight
                    if distance[v] is None or next_distance < distance[v]:
                        distance[v] = next_distance
                        heapq.heappush(queue, (next_distance, v))
            return distance

        def components():
            seen = set()
            count = 0
            for start in range(n):
                if start in seen:
                    continue
                count += 1
                stack = [start]
                while stack:
                    u = stack.pop()
                    if u in seen:
                        continue
                    seen.add(u)
                    stack.extend(v for v, _ in adjacent[u])
            return count

        output = []
        for op, u, v in queries:
            if op == "neighbors":
                values = sorted({neighbor for neighbor, _ in adjacent[u]})
                output.append(",".join(map(str, values)) if values else "-")
            elif op == "components":
                output.append(str(components()))
            else:
                distance = distances(u)[v]
                output.append(str(int(distance is not None)) if op == "reach"
                              else str(distance if distance is not None else -1))
        return ("\n".join(output) + "\n").encode()

    def correctness(self, case: Case, result) -> dict:
        return {"case": case.name, "passed": result.compiled and result.exit_code == 0 and
                not result.timed_out and result.stdout_bytes == case.expected}

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
        return f"{sum(x['passed'] for x in correctness)}/{len(correctness)} graph cases passed"

    def evaluate(self, source: str, challenge: dict, config: SandboxConfig = SandboxConfig()) -> dict:
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        started_at = _now()
        cases = self.generate_cases(challenge, challenge["seed"])
        checks, metrics, build = [], [], None
        for case in cases:
            result = run_c(source, case.stdin, config)
            if build is None:
                build = {"exit_code": result.compile_exit_code, "stdout": result.compile_stdout,
                         "stderr": result.compile_stderr}
            checks.append(self.correctness(case, result))
            metrics.append({**self.performance(case, result), "input_bytes": len(case.stdin),
                            "vertices": challenge["vertices"],
                            "within_time_limit": result.elapsed_ms <= challenge["max_ms"]})
        rewards = self.reward_inputs(checks, metrics)
        accepted = rewards["eligible_for_performance"] and all(x["within_time_limit"] for x in metrics)
        return {"started_at": started_at, "finished_at": _now(), "build": build,
                "correctness": {"passed": rewards["correctness_passed"], "total": len(checks),
                                "cases": checks},
                "performance": {"cases": metrics,
                                "median_ms": statistics.median(x["elapsed_ms"] for x in metrics),
                                "largest_graph_completed": challenge["vertices"] if accepted else 0,
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
