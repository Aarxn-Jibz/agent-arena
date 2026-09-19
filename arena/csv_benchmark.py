"""Deterministic CSV/data CLI benchmark with an independent Python oracle."""

from __future__ import annotations

import csv
import io
import random
import statistics
from datetime import datetime, timezone

from .bench_result import record_judgement
from .benchmark import Case
from .sandbox import SandboxConfig, run_c

OPERATIONS = ("select", "filter_eq", "sort", "sum", "uppercase", "count")
DISTRIBUTIONS = ("plain", "quoted", "mixed", "wide")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class CsvBenchmark:
    name = "csv"

    def initialize(self, seed: int) -> dict:
        return {"operation": "count", "rows": 20, "columns": 3, "column": 0,
                "value": "0", "distribution": "plain", "invalid": False,
                "max_ms": 3000, "repeats": 1, "seed": seed}

    def validate_challenge(self, challenge: dict) -> tuple[bool, str]:
        if not isinstance(challenge, dict) or set(challenge) != set(self.initialize(0)):
            return False, "invalid challenge fields"
        if challenge["operation"] not in OPERATIONS or challenge["distribution"] not in DISTRIBUTIONS:
            return False, "unsupported operation or distribution"
        for key, low, high in (("rows", 0, 10000), ("columns", 2, 8), ("column", 0, 7),
                               ("max_ms", 1, 10000), ("repeats", 1, 3), ("seed", 0, 2**32-1)):
            value = challenge[key]
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                return False, f"{key} out of range"
        if challenge["column"] >= challenge["columns"]:
            return False, "column not present"
        if challenge["operation"] == "sum" and challenge["column"] != 0:
            return False, "sum uses numeric column 0"
        if not isinstance(challenge["invalid"], bool):
            return False, "invalid must be boolean"
        value = challenge["value"]
        if not isinstance(value, str) or len(value) > 40 or any(ord(c) > 127 for c in value):
            return False, "value must be short ASCII text"
        return True, "valid"

    def generate_cases(self, challenge: dict, seed: int) -> list[Case]:
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        cases = []
        for repeat in range(challenge["repeats"]):
            rng = random.Random(seed + repeat)
            header = [f"c{i}" for i in range(challenge["columns"])]
            rows = [header]
            for _ in range(challenge["rows"]):
                row = [str(rng.randrange(1000))]
                for col in range(1, challenge["columns"]):
                    plain = f"v{rng.randrange(100)}"
                    special = rng.choice(("a,b", 'a"b', "two\nlines", "", "  spaces  "))
                    if challenge["distribution"] == "plain":
                        value = plain
                    elif challenge["distribution"] == "quoted":
                        value = special
                    elif challenge["distribution"] == "wide":
                        value = (plain + special) * 4
                    else:
                        value = rng.choice((plain, special))
                    row.append(value)
                rows.append(row)
            data = _csv(rows)
            if challenge["invalid"]:
                data += b'"unterminated\n'
            base = Case(f"rows-{challenge['rows']}-{repeat}", data)
            cases.append(Case(base.name, data, self.oracle(base, challenge), challenge["invalid"]))
        return cases

    def oracle(self, case: Case, challenge: dict) -> bytes | None:
        if challenge["invalid"]:
            return None  # required: nonzero exit and empty stdout
        rows = list(csv.reader(io.StringIO(case.stdin.decode("utf-8"), newline=""), strict=True))
        header, data = rows[0], rows[1:]
        op, col, value = challenge["operation"], challenge["column"], challenge["value"]
        if op == "count":
            return f"{len(data)}\n".encode()
        if op == "sum":
            return f"{sum(int(row[col]) for row in data)}\n".encode()
        if op == "select":
            return _csv([[header[col]]] + [[row[col]] for row in data])
        if op == "filter_eq":
            return _csv([header] + [row for row in data if row[col] == value])
        if op == "sort":
            return _csv([header] + sorted(data, key=lambda row: row[col]))
        if op == "uppercase":
            return _csv([header] + [[cell.upper() if i == col else cell for i, cell in enumerate(row)]
                                    for row in data])
        raise ValueError(op)

    def correctness(self, case: Case, result) -> dict:
        passed = (result.compiled and not result.timed_out and
                  ((result.exit_code != 0 and result.stdout_bytes == b"") if case.malformed
                   else (result.exit_code == 0 and result.stdout_bytes == case.expected)))
        return {"case": case.name, "passed": passed, "malformed": case.malformed}

    def performance(self, case: Case, result) -> dict:
        return {"case": case.name, "elapsed_ms": result.elapsed_ms,
                "exit_code": result.exit_code, "timed_out": result.timed_out}

    def reward_inputs(self, correctness: list[dict], performance: list[dict]) -> dict:
        passed, total = sum(x["passed"] for x in correctness), len(correctness)
        eligible = total > 0 and passed == total
        return {"correctness_passed": passed, "correctness_total": total,
                "eligible_for_performance": eligible,
                "solver_reward": (1.0 + 1.0 / (1 + statistics.median(x["elapsed_ms"] for x in performance)))
                if eligible else passed / max(1, total),
                "challenger_reward": 1 - passed / max(1, total)}

    def summarize(self, correctness: list[dict], performance: list[dict]) -> str:
        return f"{sum(x['passed'] for x in correctness)}/{len(correctness)} CSV cases passed"

    def evaluate(self, source: str, challenge: dict, config: SandboxConfig = SandboxConfig()) -> dict:
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        started_at = _now()
        checks, metrics, build = [], [], None
        for case in self.generate_cases(challenge, challenge["seed"]):
            result = run_c(source, case.stdin, config,
                           args=(challenge["operation"], str(challenge["column"]), challenge["value"]))
            if build is None:
                build = {"exit_code": result.compile_exit_code, "stdout": result.compile_stdout,
                         "stderr": result.compile_stderr}
            checks.append(self.correctness(case, result))
            metrics.append({"case": case.name, "input_bytes": len(case.stdin),
                            "output_bytes": len(result.stdout_bytes), "elapsed_ms": result.elapsed_ms,
                            "exit_code": result.exit_code, "timed_out": result.timed_out,
                            "within_time_limit": result.elapsed_ms <= challenge["max_ms"],
                            "rows": challenge["rows"]})
        rewards = self.reward_inputs(checks, metrics)
        within_limit = all(x["within_time_limit"] for x in metrics)
        accepted = rewards["eligible_for_performance"] and within_limit
        return {"started_at": started_at, "finished_at": _now(), "build": build,
                "correctness": {"passed": rewards["correctness_passed"], "total": len(checks),
                                "cases": checks},
                "performance": {"cases": metrics,
                                "median_ms": statistics.median(x["elapsed_ms"] for x in metrics),
                                "largest_workload_completed": max((x["rows"] for x, c in zip(metrics, checks)
                                                                    if c["passed"]), default=0),
                                "resource_status": "within limits" if within_limit else "time limit exceeded",
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
