"""Local Unix-socket HTTP benchmark; candidate remains in networkless Docker."""

from __future__ import annotations

import base64
import json
import random
import statistics
from datetime import datetime, timezone

from .bench_result import record_judgement
from .sandbox import SandboxConfig, run_c

ROUTES = ("health", "echo", "missing")


def _now():
    return datetime.now(timezone.utc).isoformat()


class HttpBenchmark:
    name = "http"

    def initialize(self, seed: int) -> dict:
        return {"routes": ["health", "echo", "missing"], "requests": 6,
                "body_bytes": 32, "concurrency": 1, "malformed": False,
                "partial": False, "max_ms": 3000, "seed": seed}

    def validate_challenge(self, challenge: dict) -> tuple[bool, str]:
        if not isinstance(challenge, dict) or set(challenge) != set(self.initialize(0)):
            return False, "invalid challenge fields"
        routes = challenge["routes"]
        if not isinstance(routes, list) or not routes or len(routes) > 3 or len(set(map(str, routes))) != len(routes) or any(x not in ROUTES for x in routes):
            return False, "invalid routes"
        for key in ("malformed", "partial"):
            if type(challenge[key]) is not bool:
                return False, f"{key} must be boolean"
        for key, low, high in (("requests", 1, 20), ("body_bytes", 0, 512),
                               ("concurrency", 1, 8), ("max_ms", 1, 10000),
                               ("seed", 0, 2**32-1)):
            value = challenge[key]
            if type(value) is not int or not low <= value <= high:
                return False, f"{key} out of range"
        return True, "valid"

    def generate_requests(self, challenge: dict) -> list[dict]:
        valid, reason = self.validate_challenge(challenge)
        if not valid:
            raise ValueError(reason)
        rng = random.Random(challenge["seed"])
        cases = []
        for i in range(challenge["requests"]):
            route = rng.choice(challenge["routes"])
            body = bytes(rng.randrange(256) for _ in range(challenge["body_bytes"])) if route == "echo" else b""
            method, path = ("POST", "/echo") if route == "echo" else ("GET", "/health" if route == "health" else "/missing")
            raw = (f"{method} {path} HTTP/1.1\r\nHost: local\r\n"
                   f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
            expected = (200, body if route == "echo" else (b"ok\n" if route == "health" else b"not found\n"),
                        "application/octet-stream" if route == "echo" else "text/plain")
            cases.append({"name": f"request-{i}", "raw": raw, "expected": expected})
        if challenge["malformed"]:
            cases.append({"name": "malformed", "raw": b"GARBAGE\r\n\r\n",
                          "expected": (400, b"bad request\n", "text/plain")})
        if challenge["partial"]:
            cases.append({"name": "partial", "raw": b"GET /health HTTP/1.1\r\nHost: local\r\n",
                          "expected": (400, b"bad request\n", "text/plain")})
        return cases

    @staticmethod
    def parse_response(raw: bytes):
        try:
            headers_raw, body = raw.split(b"\r\n\r\n", 1)
            lines = headers_raw.decode("ascii").split("\r\n")
            version, status, _ = lines[0].split(" ", 2)
            if version != "HTTP/1.1":
                return None
            headers = dict(line.split(":", 1) for line in lines[1:])
            headers = {key.lower(): value.strip().lower() for key, value in headers.items()}
            if int(headers["content-length"]) != len(body):
                return None
            return int(status), body, headers.get("content-type")
        except (ValueError, KeyError, UnicodeError):
            return None

    def correctness(self, case: dict, response: dict) -> dict:
        parsed = self.parse_response(base64.b64decode(response["response_b64"]))
        return {"case": case["name"], "passed": response["error"] is None and parsed == case["expected"],
                "error": response["error"]}

    def performance(self, case: dict, response: dict) -> dict:
        return {"case": case["name"], "latency_ms": response["latency_ms"], "error": response["error"]}

    def reward_inputs(self, correctness: list[dict], performance: list[dict]) -> dict:
        passed, total = sum(x["passed"] for x in correctness), len(correctness)
        eligible = total > 0 and passed == total
        return {"correctness_passed": passed, "correctness_total": total,
                "eligible_for_performance": eligible,
                "solver_reward": 1 + 1 / (1 + statistics.median(x["latency_ms"] for x in performance))
                if eligible else passed / max(1, total),
                "challenger_reward": 1 - passed / max(1, total)}

    def summarize(self, correctness: list[dict], performance: list[dict]) -> str:
        return f"{sum(x['passed'] for x in correctness)}/{len(correctness)} HTTP requests passed"

    def evaluate(self, source: str, challenge: dict, config: SandboxConfig = SandboxConfig()) -> dict:
        cases = self.generate_requests(challenge)
        started_at = _now()
        payload = json.dumps({"requests": [base64.b64encode(case["raw"]).decode() for case in cases],
                              "concurrency": challenge["concurrency"]}).encode()
        result = run_c(source, payload, config, args=("--http",))
        build = {"exit_code": result.compile_exit_code, "stdout": result.compile_stdout,
                 "stderr": result.compile_stderr}
        if result.compiled and result.exit_code == 0:
            try:
                measured = json.loads(result.stdout_bytes)
                responses = measured["responses"]
                if len(responses) != len(cases):
                    raise ValueError("response count mismatch")
            except (ValueError, KeyError, TypeError):
                measured, responses = {"elapsed_ms": result.elapsed_ms, "server_exit": None}, []
        else:
            measured, responses = {"elapsed_ms": result.elapsed_ms, "server_exit": result.exit_code}, []
        if responses:
            checks = [self.correctness(case, response) for case, response in zip(cases, responses)]
            metrics = [self.performance(case, response) for case, response in zip(cases, responses)]
        else:
            checks = [{"case": case["name"], "passed": False, "error": "server unavailable"} for case in cases]
            metrics = [{"case": case["name"], "latency_ms": result.elapsed_ms,
                        "error": "server unavailable"} for case in cases]
        rewards = self.reward_inputs(checks, metrics)
        latencies = sorted(x["latency_ms"] for x in metrics)
        elapsed = measured["elapsed_ms"]
        accepted = rewards["eligible_for_performance"] and elapsed <= challenge["max_ms"]
        return {"started_at": started_at, "finished_at": _now(), "build": build,
                "correctness": {"passed": rewards["correctness_passed"], "total": len(cases),
                                "cases": checks},
                "performance": {"cases": metrics, "median_latency_ms": statistics.median(latencies),
                                "p95_latency_ms": latencies[max(0, (95 * len(latencies) + 99) // 100 - 1)],
                                "throughput_per_second": len(cases) * 1000 / elapsed if elapsed else None,
                                "elapsed_ms": elapsed, "server_exit": measured["server_exit"],
                                "resource_status": "within limits" if accepted else "failed or limit exceeded",
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
