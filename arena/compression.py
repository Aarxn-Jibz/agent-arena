"""Deterministic lossless-compression benchmark. Candidate code runs only in Docker."""

from __future__ import annotations

import hashlib
import random
import statistics
from datetime import datetime, timezone

from .benchmark import Case
from .bench_result import record_judgement
from .sandbox import SandboxConfig, run_c

KINDS = ("repetitive", "text", "source", "records", "mixed", "entropy", "binary", "edge")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CompressionBenchmark:
    name = "compression"

    def initialize(self, seed: int) -> dict:
        return {"distributions": list(KINDS), "size": 256, "seed": seed, "repeats": 1}

    def validate_challenge(self, challenge: dict) -> tuple[bool, str]:
        if not isinstance(challenge, dict):
            return False, "challenge must be an object"
        if set(challenge) != {"distributions", "size", "seed", "repeats"}:
            return False, "expected distributions, size, seed, repeats"
        kinds = challenge["distributions"]
        if not isinstance(kinds, list) or not kinds or len(kinds) > len(KINDS) or len(set(map(str, kinds))) != len(kinds) or any(k not in KINDS for k in kinds):
            return False, "invalid distributions"
        for key, low, high in (("size", 0, 8192), ("seed", 0, 2**32 - 1), ("repeats", 1, 3)):
            value = challenge[key]
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                return False, f"{key} out of range"
        return True, "valid"

    def generate_cases(self, challenge: dict, seed: int) -> list[Case]:
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        return [Case(f"{kind}-{repeat}", self._generate(kind, challenge["size"], seed + repeat))
                for kind in challenge["distributions"] for repeat in range(challenge["repeats"])]

    @staticmethod
    def _generate(kind: str, size: int, seed: int) -> bytes:
        rng = random.Random(seed)
        if kind == "edge":
            return bytes((0, 255, 10, 0, 1)[:size]) if size <= 5 else bytes((0, 255, 10, 0, 1)) + bytes(size - 5)
        if kind == "repetitive":
            return (b"ABCD" * ((size + 3) // 4))[:size]
        if kind == "text":
            words = (b"quiet", b"river", b"under", b"clouds", b"moves", b"again")
            return (b" ".join(rng.choice(words) for _ in range(size // 3 + 2)))[:size]
        if kind == "source":
            lines = (b"int value = 0;\n", b"value += 1;\n", b"return value;\n")
            return (b"".join(rng.choice(lines) for _ in range(size // 8 + 2)))[:size]
        if kind == "records":
            rows = [f"{i},{rng.randrange(1000)},item_{rng.randrange(20)}\n".encode()
                    for i in range(size // 8 + 2)]
            return b"".join(rows)[:size]
        if kind == "mixed":
            blocks = (b"A" * 29, b"\x00\xff\x01", b"field,value\n", bytes(range(32)))
            return (b"".join(rng.choice(blocks) for _ in range(size // 3 + 2)))[:size]
        if kind == "entropy":
            return bytes(rng.randrange(256) for _ in range(size))
        if kind == "binary":
            return bytes(rng.choice((0, 0, 255, 127, 10, 13, rng.randrange(256))) for _ in range(size))
        raise ValueError(kind)

    def correctness(self, case: Case, result) -> dict:
        return {"case": case.name, "passed": result.compiled and result.exit_code == 0 and
                not result.timed_out and result.stdout_bytes == case.stdin}

    def performance(self, case: Case, result) -> dict:
        return {"case": case.name, "elapsed_ms": result.elapsed_ms,
                "exit_code": result.exit_code, "timed_out": result.timed_out}

    def reward_inputs(self, correctness: list[dict], performance: list[dict]) -> dict:
        passed = sum(item["passed"] for item in correctness)
        total = len(correctness)
        valid = total > 0 and passed == total
        savings = statistics.mean(item["bytes_saved"] / max(1, item["original_bytes"])
                                  for item in performance) if valid else 0.0
        return {"correctness_passed": passed, "correctness_total": total,
                "eligible_for_performance": valid, "mean_savings_fraction": savings,
                "solver_reward": (1.0 + max(0.0, savings)) if valid else passed / max(1, total),
                "challenger_reward": 1.0 - passed / max(1, total)}

    def summarize(self, correctness: list[dict], performance: list[dict]) -> str:
        passed = sum(x["passed"] for x in correctness)
        return f"{passed}/{len(correctness)} round trips passed; {len(performance)} cases measured"

    def evaluate(self, source: str, challenge: dict,
                 config: SandboxConfig = SandboxConfig()) -> dict:
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        started_at = _now()
        checks, metrics = [], []
        build = None
        for case in self.generate_cases(challenge, challenge["seed"]):
            original = case.stdin
            compressed = run_c(source, original, config, args=("compress",))
            if build is None:
                build = {"exit_code": compressed.compile_exit_code,
                         "stdout": compressed.compile_stdout, "stderr": compressed.compile_stderr}
            expanded = (run_c(source, compressed.stdout_bytes, config, args=("decompress",))
                        if compressed.compiled and compressed.exit_code == 0 and not compressed.timed_out
                        else None)
            passed = bool(expanded and expanded.compiled and expanded.exit_code == 0 and
                          not expanded.timed_out and expanded.stdout_bytes == original)
            checks.append({"case": case.name, "passed": passed,
                           "input_sha256": hashlib.sha256(original).hexdigest(),
                           "roundtrip_sha256": hashlib.sha256(expanded.stdout_bytes).hexdigest() if expanded else None})
            compressed_size = len(compressed.stdout_bytes)
            metrics.append({"case": case.name, "original_bytes": len(original),
                            "compressed_bytes": compressed_size,
                            "ratio": compressed_size / len(original) if original else None,
                            "bytes_saved": len(original) - compressed_size,
                            "compression_ms": compressed.elapsed_ms,
                            "decompression_ms": expanded.elapsed_ms if expanded else None,
                            "resource_status": {"compress_exit": compressed.exit_code,
                                                "decompress_exit": expanded.exit_code if expanded else None,
                                                "compress_timeout": compressed.timed_out,
                                                "decompress_timeout": expanded.timed_out if expanded else None}})
        rewards = self.reward_inputs(checks, metrics)
        return {"started_at": started_at, "finished_at": _now(),
                "build": build, "correctness": {"passed": rewards["correctness_passed"],
                                                  "total": rewards["correctness_total"], "cases": checks},
                "performance": {"cases": metrics,
                                "compression_median_ms": statistics.median(x["compression_ms"] for x in metrics),
                                "decompression_median_ms": statistics.median(x["decompression_ms"] for x in metrics if x["decompression_ms"] is not None) if any(x["decompression_ms"] is not None for x in metrics) else None},
                "reward_inputs": rewards, "feedback": self.summarize(checks, metrics),
                "accepted": rewards["eligible_for_performance"]}

    def record_episode(self, root, *, run_id: str, episode_id: int, challenge: dict,
                       source: str, git_before: str, git_after: str | None,
                       solver_response: str = "", rationale: str | None = None,
                       patch: str | None = None, config: SandboxConfig = SandboxConfig(),
                       evaluated: dict | None = None):
        result = evaluated if evaluated is not None else self.evaluate(source, challenge, config)
        return record_judgement(root, benchmark=self.name, run_id=run_id,
                                episode_id=episode_id, challenge=challenge, source=source,
                                evaluation=result, git_before=git_before, git_after=git_after,
                                solver_response=solver_response, rationale=rationale, patch=patch)
