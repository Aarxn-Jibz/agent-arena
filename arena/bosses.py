"""Held-out boss Judges. Never imported by training task selection."""

from __future__ import annotations

import json
import os
import random
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .sandbox import SandboxConfig, run_c


def _now():
    return datetime.now(timezone.utc).isoformat()


class MiniShellBoss:
    name = "mini_shell"

    def initialize(self, seed):
        return {"commands": 12, "seed": seed, "max_ms": 3000}

    def validate_challenge(self, challenge):
        return (isinstance(challenge, dict) and set(challenge) == {"commands", "seed", "max_ms"}
                and all(type(challenge[key]) is int and low <= challenge[key] <= high
                        for key, low, high in (("commands", 1, 50), ("seed", 0, 2**32-1),
                                               ("max_ms", 1, 10000))))

    def generate(self, challenge):
        if not self.validate_challenge(challenge):
            raise ValueError("invalid mini-shell challenge")
        rng = random.Random(challenge["seed"])
        vocabulary = ("echo hello", "run true", "run false", "run echo child",
                      "status", "pwd", "cd /work", "cd /", "set name value", "get name")
        return [rng.choice(vocabulary) for _ in range(challenge["commands"])]

    @staticmethod
    def oracle(commands):
        variables, cwd, last, output = {}, "/", 0, []
        exit_code = 0
        for line in commands:
            if line.startswith("echo "):
                output.append(line[5:])
            elif line == "run true":
                last = 0
            elif line == "run false":
                last = 1
            elif line.startswith("run echo "):
                output.append(line[9:])
                last = 0
            elif line == "status":
                output.append(str(last))
            elif line == "pwd":
                output.append(cwd)
            elif line in ("cd /", "cd /work"):
                cwd = line[3:]
            elif line.startswith("set "):
                _, key, value = line.split(" ", 2)
                variables[key] = value
            elif line.startswith("get "):
                output.append(variables.get(line[4:], ""))
            elif line.startswith("exit "):
                exit_code = int(line[5:])
                break
            else:
                output.append("ERROR")
        return ("\n".join(output) + ("\n" if output else "")).encode(), exit_code

    def evaluate(self, source, challenge, config=SandboxConfig()):
        commands = self.generate(challenge)
        expected, exit_code = self.oracle(commands)
        result = run_c(source, ("\n".join(commands) + "\n").encode(), config)
        passed = result.compiled and result.exit_code == exit_code and result.stdout_bytes == expected
        return {"passed": passed, "elapsed_ms": result.elapsed_ms,
                "within_limit": result.elapsed_ms <= challenge["max_ms"],
                "build": {"exit_code": result.compile_exit_code, "stderr": result.compile_stderr},
                "resource_status": {"timed_out": result.timed_out, "exit_code": result.exit_code}}


class TinyFilesystemBoss:
    name = "tiny_filesystem"
    IMAGE_BYTES = 4096

    def initialize(self, seed):
        return {"operations": 10, "seed": seed, "max_ms": 3000}

    def validate_challenge(self, challenge):
        return (isinstance(challenge, dict) and set(challenge) == {"operations", "seed", "max_ms"}
                and all(type(challenge[key]) is int and low <= challenge[key] <= high
                        for key, low, high in (("operations", 1, 30), ("seed", 0, 2**32-1),
                                               ("max_ms", 1, 10000))))

    def generate(self, challenge):
        if not self.validate_challenge(challenge):
            raise ValueError("invalid tiny-filesystem challenge")
        rng = random.Random(challenge["seed"])
        names = ("/a", "/b", "/c", "/d")
        commands = []
        for _ in range(challenge["operations"]):
            name = rng.choice(names)
            op = rng.choice(("CREATE", "WRITE", "READ", "DELETE", "LIST"))
            commands.append(f"{op} {name} {rng.randrange(256):02x}" if op == "WRITE"
                            else f"{op} {name}" if op != "LIST" else "LIST")
        return commands

    @staticmethod
    def oracle_step(files, command):
        parts = command.split()
        op = parts[0]
        if op == "LIST":
            return " ".join(sorted(files)) + "\n"
        name = parts[1]
        if op == "CREATE":
            if name in files or len(files) >= 32:
                return "ERR\n"
            files[name] = b""
            return "OK\n"
        if op == "DELETE":
            if name not in files:
                return "ERR\n"
            del files[name]
            return "OK\n"
        if op == "READ":
            return f"DATA {files[name].hex()}\n" if name in files else "ERR\n"
        if op == "WRITE" and name in files:
            content = bytes.fromhex(parts[2])
            if sum(map(len, files.values())) - len(files[name]) + len(content) > 2048:
                return "ERR\n"
            files[name] = content
            return "OK\n"
        return "ERR\n"

    def evaluate(self, source, challenge, config=SandboxConfig()):
        commands = self.generate(challenge)
        image = bytes(self.IMAGE_BYTES)
        files = {}
        checks, times = [], []
        build = None
        for command in commands:
            expected = self.oracle_step(files, command).encode()
            result = run_c(source, image + command.encode() + b"\n", config, args=("apply",))
            if build is None:
                build = {"exit_code": result.compile_exit_code, "stderr": result.compile_stderr}
            valid = (result.compiled and result.exit_code == 0 and not result.timed_out and
                     len(result.stdout_bytes) >= self.IMAGE_BYTES and
                     result.stdout_bytes[self.IMAGE_BYTES:] == expected)
            checks.append(valid)
            times.append(result.elapsed_ms)
            if valid:
                image = result.stdout_bytes[:self.IMAGE_BYTES]
            else:
                break
        return {"passed": len(checks) == len(commands) and all(checks),
                "steps_passed": sum(checks), "steps_total": len(commands),
                "median_ms": statistics.median(times) if times else None,
                "within_limit": all(t <= challenge["max_ms"] for t in times),
                "build": build, "resource_status": {"last_step_passed": checks[-1] if checks else False}}


def freeze_zero_shot(root, boss, challenge, *, baseline_source, trained_source,
                     baseline_commit, trained_commit, config=SandboxConfig()):
    """Evaluate both candidates before publishing one immutable first-use result."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{boss.name}-zero-shot.json"
    if path.exists():
        raise FileExistsError("first zero-shot result already frozen")
    if not boss.validate_challenge(challenge):
        raise ValueError("invalid held-out challenge")
    baseline = boss.evaluate(baseline_source, challenge, config)
    trained = boss.evaluate(trained_source, challenge, config)
    result = {"benchmark": boss.name, "phase": "zero_shot", "challenge": challenge,
              "baseline_commit": baseline_commit, "trained_commit": trained_commit,
              "baseline": baseline, "trained": trained, "frozen_at": _now()}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return path
