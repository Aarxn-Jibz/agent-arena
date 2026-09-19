"""Validation benchmarks: cache, log analysis, and document search."""

from __future__ import annotations

import random
import statistics
from dataclasses import replace
from datetime import datetime, timezone

from .bench_result import record_judgement
from .sandbox import SandboxConfig, run_c


def _now():
    return datetime.now(timezone.utc).isoformat()


def _integer(challenge, key, low, high):
    value = challenge[key]
    return type(value) is int and low <= value <= high


class ValidationBenchmark:
    """Small common Judge shell; subclasses own challenge and output semantics."""

    def record_episode(self, root, *, run_id, episode_id, challenge, source, evaluation,
                       git_before, git_after, solver_response="", rationale=None, patch=None):
        return record_judgement(root, benchmark=self.name, run_id=run_id,
                                episode_id=episode_id, challenge=challenge, source=source,
                                evaluation=evaluation, git_before=git_before, git_after=git_after,
                                solver_response=solver_response, rationale=rationale, patch=patch)

    def _result(self, started, source, challenge, checks, timings, build, extra=None):
        passed, total = sum(checks), len(checks)
        correct = passed == total and total > 0
        median = statistics.median(timings) if timings else None
        rewards = {"correctness_passed": passed, "correctness_total": total,
                   "eligible_for_performance": correct,
                   "solver_reward": 1 + 1 / (1 + median) if correct else passed / max(1, total),
                   "challenger_reward": 1 - passed / max(1, total)}
        return {"started_at": started, "finished_at": _now(), "build": build,
                "correctness": {"passed": passed, "total": total, "cases": checks},
                "performance": {"median_ms": median, "case_ms": timings,
                                "memory_bytes": None, **(extra or {})},
                "reward_inputs": rewards, "feedback": f"{passed}/{total} cases passed",
                "accepted": correct and all(t <= challenge["max_ms"] for t in timings)}


class CacheBenchmark(ValidationBenchmark):
    name = "cache"

    def initialize(self, seed):
        return {"capacity": 4, "operations": 30, "keys": 8,
                "distribution": "uniform", "max_ms": 3000, "seed": seed}

    def validate_challenge(self, challenge):
        if not isinstance(challenge, dict) or set(challenge) != set(self.initialize(0)):
            return False, "invalid fields"
        if challenge["distribution"] not in ("uniform", "hot", "sequential"):
            return False, "invalid distribution"
        if not all((_integer(challenge, "capacity", 1, 128),
                    _integer(challenge, "operations", 1, 1000),
                    _integer(challenge, "keys", 1, 256),
                    _integer(challenge, "max_ms", 1, 10000),
                    _integer(challenge, "seed", 0, 2**32-1))):
            return False, "numeric field out of range"
        return True, "valid"

    def generate_trace(self, challenge):
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        rng = random.Random(challenge["seed"])
        trace = []
        for i in range(challenge["operations"]):
            key = (i % challenge["keys"] if challenge["distribution"] == "sequential" else
                   rng.randrange(min(3, challenge["keys"])) if challenge["distribution"] == "hot" and rng.randrange(4) else
                   rng.randrange(challenge["keys"]))
            trace.append(("PUT", key, rng.randrange(10000)) if i < challenge["capacity"] or rng.randrange(3) == 0
                         else ("GET", key, None))
        return trace

    @staticmethod
    def check(trace, lines, capacity):
        if len(lines) != len(trace):
            return False, 0, 0
        state = {}
        hits = misses = 0
        for (op, key, value), output in zip(trace, lines):
            if op == "GET":
                expected = f"HIT {state[key]}" if key in state else "MISS"
                if output != expected:
                    return False, hits, misses
                hits += key in state
                misses += key not in state
            elif key in state or len(state) < capacity:
                if output != "STORED":
                    return False, hits, misses
                state[key] = value
            else:
                parts = output.split()
                if len(parts) != 2 or parts[0] != "EVICT" or not parts[1].isdigit() or int(parts[1]) not in state:
                    return False, hits, misses
                del state[int(parts[1])]
                state[key] = value
        return True, hits, misses

    def evaluate(self, source, challenge, config=SandboxConfig()):
        trace = self.generate_trace(challenge)
        payload = (f"{challenge['capacity']} {len(trace)}\n" + "".join(
            f"{op} {key}" + (f" {value}" if value is not None else "") + "\n"
            for op, key, value in trace)).encode()
        started = _now()
        result = run_c(source, payload, config)
        lines = result.stdout_bytes.decode("utf-8", errors="replace").splitlines()
        valid, hits, misses = self.check(trace, lines, challenge["capacity"])
        valid = bool(result.compiled and result.exit_code == 0 and not result.timed_out and valid)
        evaluated = self._result(started, source, challenge, [valid], [result.elapsed_ms],
                                 {"exit_code": result.compile_exit_code, "stdout": result.compile_stdout,
                                  "stderr": result.compile_stderr},
                                 {"hits": hits, "misses": misses,
                                  "hit_rate": hits / (hits + misses) if hits + misses else None,
                                  "operations": len(trace), "throughput_per_second":
                                  len(trace) * 1000 / result.elapsed_ms if result.elapsed_ms else None,
                                  "resource_status": {"exit_code": result.exit_code,
                                                      "timed_out": result.timed_out}})
        return evaluated


class LogBenchmark(ValidationBenchmark):
    name = "log"

    def initialize(self, seed):
        return {"records": 100, "distribution": "uniform", "query": "count",
                "level": "ERROR", "source": "svc0", "malformed": 0,
                "max_ms": 3000, "seed": seed}

    def validate_challenge(self, challenge):
        if not isinstance(challenge, dict) or set(challenge) != set(self.initialize(0)):
            return False, "invalid fields"
        if challenge["distribution"] not in ("uniform", "bursty") or challenge["query"] not in ("count", "by_level", "filter"):
            return False, "unsupported distribution or query"
        if challenge["level"] not in ("INFO", "WARN", "ERROR") or challenge["source"] not in ("svc0", "svc1", "svc2"):
            return False, "unsupported filter"
        if not all((_integer(challenge, "records", 0, 10000), _integer(challenge, "malformed", 0, 100),
                    _integer(challenge, "max_ms", 1, 10000), _integer(challenge, "seed", 0, 2**32-1))):
            return False, "numeric field out of range"
        return True, "valid"

    def generate(self, challenge):
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        rng = random.Random(challenge["seed"])
        rows = []
        for i in range(challenge["records"]):
            level = rng.choice(("INFO", "WARN", "ERROR")) if challenge["distribution"] == "uniform" else (
                "ERROR" if i % 20 < 15 else "INFO")
            rows.append(f"{i:08d}|{level}|svc{rng.randrange(3)}|msg{rng.randrange(100)}")
        rows += ["INVALID"] * challenge["malformed"]
        return rows

    def oracle(self, rows, challenge):
        valid = [line.split("|") for line in rows if len(line.split("|")) == 4]
        if challenge["query"] == "count":
            return f"{sum(x[1] == challenge['level'] for x in valid)}\n".encode()
        if challenge["query"] == "by_level":
            return ("\n".join(f"{level} {sum(x[1] == level for x in valid)}"
                              for level in ("INFO", "WARN", "ERROR")) + "\n").encode()
        return ("\n".join("|".join(x) for x in valid if x[1] == challenge["level"] and
                          x[2] == challenge["source"]) + ("\n" if any(
                              x[1] == challenge["level"] and x[2] == challenge["source"] for x in valid) else "")).encode()

    def evaluate(self, source, challenge, config=SandboxConfig()):
        rows = self.generate(challenge)
        started = _now()
        result = run_c(source, ("\n".join(rows) + "\n").encode(), config,
                       args=(challenge["query"], challenge["level"], challenge["source"]))
        expected = self.oracle(rows, challenge)
        passed = bool(result.compiled and result.exit_code == 0 and not result.timed_out and result.stdout_bytes == expected)
        return self._result(started, source, challenge, [passed], [result.elapsed_ms],
                            {"exit_code": result.compile_exit_code, "stdout": result.compile_stdout,
                             "stderr": result.compile_stderr},
                            {"records": len(rows), "input_bytes": sum(len(x) + 1 for x in rows),
                             "throughput_per_second": len(rows) * 1000 / result.elapsed_ms if result.elapsed_ms else None,
                             "resource_status": {"exit_code": result.exit_code,
                                                 "timed_out": result.timed_out}})


class SearchBenchmark(ValidationBenchmark):
    name = "search"

    def initialize(self, seed):
        return {"documents": 20, "queries": 5, "words_per_doc": 10,
                "distribution": "uniform", "matching": "exact", "max_ms": 3000,
                "seed": seed}

    def validate_challenge(self, challenge):
        if not isinstance(challenge, dict) or set(challenge) != set(self.initialize(0)):
            return False, "invalid fields"
        if challenge["distribution"] not in ("uniform", "skewed") or challenge["matching"] not in ("exact", "prefix"):
            return False, "unsupported distribution or matching"
        if not all((_integer(challenge, "documents", 0, 1000), _integer(challenge, "queries", 1, 30),
                    _integer(challenge, "words_per_doc", 1, 100), _integer(challenge, "max_ms", 1, 10000),
                    _integer(challenge, "seed", 0, 2**32-1))):
            return False, "numeric field out of range"
        return True, "valid"

    def generate(self, challenge):
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        rng = random.Random(challenge["seed"])
        vocabulary = [f"w{i}" for i in range(40)]
        docs = []
        for _ in range(challenge["documents"]):
            docs.append([rng.choice(vocabulary[:5] if challenge["distribution"] == "skewed"
                                    and rng.randrange(4) else vocabulary)
                         for _ in range(challenge["words_per_doc"])])
        queries = [rng.choice(vocabulary) for _ in range(challenge["queries"])]
        return docs, queries

    @staticmethod
    def oracle(docs, query, matching):
        return (" ".join(str(i) for i, words in enumerate(docs)
                         if any((word == query if matching == "exact" else word.startswith(query))
                                for word in words)) + "\n").encode()

    def evaluate(self, source, challenge, config=SandboxConfig()):
        docs, queries = self.generate(challenge)
        started = _now()
        document_input = (str(len(docs)) + "\n" + "\n".join(" ".join(words) for words in docs) + "\n").encode()
        index_config = replace(config, output_bytes=min(8 * 1048576,
            max(config.output_bytes, 1024 + 2 * len(document_input))))
        built = run_c(source, document_input, index_config, args=("build",))
        build = {"exit_code": built.compile_exit_code, "stdout": built.compile_stdout,
                 "stderr": built.compile_stderr}
        checks, query_times = [], []
        if built.compiled and built.exit_code == 0 and not built.timed_out:
            for query in queries:
                index = built.stdout_bytes
                payload = len(index).to_bytes(4, "big") + index + query.encode() + b"\n"
                result = run_c(source, payload, index_config, args=("query", challenge["matching"]))
                checks.append(result.compiled and result.exit_code == 0 and not result.timed_out and
                              result.stdout_bytes == self.oracle(docs, query, challenge["matching"]))
                query_times.append(result.elapsed_ms)
        else:
            checks = [False] * len(queries)
            query_times = [built.elapsed_ms] * len(queries)
        evaluated = self._result(started, source, challenge, checks,
                                 [built.elapsed_ms, *query_times], build,
                                 {"index_bytes": len(built.stdout_bytes), "indexing_ms": built.elapsed_ms,
                                  "query_median_ms": statistics.median(query_times),
                                  "documents": len(docs), "resource_status":
                                  {"build_exit": built.exit_code, "build_timeout": built.timed_out}})
        return evaluated
