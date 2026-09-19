"""Benchmark contract. A benchmark specifies requirements and measurements only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .sandbox import SandboxResult


@dataclass(frozen=True)
class Case:
    """One deterministic generated input; keep its generator config in evidence."""

    name: str
    stdin: str | bytes
    expected: bytes | None = None
    malformed: bool = False


class Benchmark(Protocol):
    name: str

    def initialize(self, seed: int) -> dict: ...

    def validate_challenge(self, challenge: dict) -> tuple[bool, str]: ...

    def generate_cases(self, challenge: dict, seed: int) -> list[Case]: ...

    def correctness(self, case: Case, result: SandboxResult) -> dict: ...

    def performance(self, case: Case, result: SandboxResult) -> dict: ...

    def reward_inputs(self, correctness: list[dict], performance: list[dict]) -> dict: ...

    def summarize(self, correctness: list[dict], performance: list[dict]) -> str: ...
