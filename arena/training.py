"""Runtime-independent training architecture for the GPU experiment.

This module deliberately uses only the standard library.  It is the contract
between the arena and a future HF/PEFT runtime; tests use ``MockTrainer``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


PRODUCTION_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct-16k"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = PRODUCTION_MODEL
    revision: str | None = None
    tokenizer_revision: str | None = None
    context_length: int = 16384
    precision: str = "bf16"
    quantization: str = "none"
    lora: dict[str, Any] = field(default_factory=lambda: {"r": 16, "alpha": 32, "dropout": 0.05})
    generation: dict[str, Any] = field(default_factory=lambda: {"max_new_tokens": 2048, "temperature": 0.7, "top_p": 0.9})


def production_model_config() -> ModelConfig:
    return ModelConfig()


def _version(name: str) -> str | None:
    try:
        return __import__(name).__version__
    except (ImportError, AttributeError):
        return None


def hardware_profile() -> dict[str, Any]:
    """Read-only doctor probe; imports optional ML packages only when present."""
    disk = shutil.disk_usage(Path.cwd())
    profile = {"os": platform.platform(), "python": sys.version.split()[0],
               "cpu": platform.processor() or platform.machine(), "ram_gb": None,
               "free_disk_gb": round(disk.free / 2**30, 2), "torch": _version("torch"),
               "transformers": _version("transformers"), "peft": _version("peft"),
               "bitsandbytes": _version("bitsandbytes"), "cuda": False, "cuda_version": None,
               "gpu_name": None, "vram_gb": 0, "bf16": False, "fp16": False,
               "docker": shutil.which("docker") is not None, "wsl": "microsoft" in platform.release().lower(),
               "compiler": shutil.which("tcc") or shutil.which("gcc") or shutil.which("clang"),
               "model_cached": False}
    try:
        import torch
        profile["cuda"] = bool(torch.cuda.is_available())
        profile["cuda_version"] = torch.version.cuda
        if profile["cuda"]:
            p = torch.cuda.get_device_properties(0)
            profile.update(gpu_name=p.name, vram_gb=round(p.total_memory / 2**30, 2),
                           bf16=bool(torch.cuda.is_bf16_supported()), fp16=True)
    except ImportError:
        pass
    try:
        import os as _os
        profile["ram_gb"] = round(_os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
    except (AttributeError, ValueError, OSError):
        pass
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    profile["model_cache"] = str(cache)
    profile["model_cached"] = cache.exists() and any(cache.rglob("*SmolLM2*"))
    return profile


def doctor_recommendations(profile: dict[str, Any]) -> dict[str, Any]:
    vram = float(profile.get("vram_gb") or 0)
    if not profile.get("cuda"):
        return {"precision": "cpu/no training", "adaptation": "LoRA", "batch_size": 1,
                "gradient_accumulation": 16, "solver_attempts": 2,
                "note": "No CUDA GPU detected; use only mocks/CPU smoke tests."}
    if vram <= 16:
        return {"precision": "fp16", "adaptation": "QLoRA", "batch_size": 1,
                "gradient_accumulation": 16, "solver_attempts": 2}
    if vram <= 32:
        return {"precision": "bf16" if profile.get("bf16") else "fp16", "adaptation": "LoRA",
                "batch_size": 1, "gradient_accumulation": 8, "solver_attempts": 3}
    return {"precision": "bf16" if profile.get("bf16") else "fp16", "adaptation": "LoRA",
            "batch_size": 2, "gradient_accumulation": 4, "solver_attempts": 3}


def doctor_report(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = hardware_profile() if profile is None else profile
    return {"hardware": profile, "recommendations": doctor_recommendations(profile),
            "windows_judge": "UNVERIFIED" if platform.system() != "Windows" else "requires smoke test"}


@dataclass
class BenchmarkProgress:
    attempts: int = 0; first_success: int = 0; final_success: int = 0; partial_total: float = 0.0
    compile_failures: int = 0; failures: int = 0; recent: list[float] = field(default_factory=list)
    frontier: int = 1; mastered: list[int] = field(default_factory=list); unlocked: list[int] = field(default_factory=lambda: [1])


class Curriculum:
    """Deterministic curriculum: 3 >=80% final passes at a level unlocks next."""
    def __init__(self, benchmarks: list[str], max_difficulty: int = 8, mastery_attempts: int = 3, mastery_rate: float = .8):
        self.max_difficulty, self.mastery_attempts, self.mastery_rate = max_difficulty, mastery_attempts, mastery_rate
        self.progress = {name: BenchmarkProgress() for name in benchmarks}
        self.by_level = {name: {i: [] for i in range(1, max_difficulty + 1)} for name in benchmarks}

    def legal_actions(self, benchmark: str) -> list[dict[str, Any]]:
        p = self.progress[benchmark]
        levels = sorted(set(p.unlocked + [max(1, p.frontier - 1)]))
        return [{"benchmark": benchmark, "difficulty": d} for d in levels if d <= self.max_difficulty]

    def validate(self, action: Any) -> tuple[bool, str]:
        if not isinstance(action, dict) or set(action) - {"benchmark", "difficulty", "distribution", "size_band"}:
            return False, "action is outside curriculum schema"
        name, level = action.get("benchmark"), action.get("difficulty")
        if name not in self.progress or type(level) is not int:
            return False, "unknown benchmark or difficulty"
        if level not in [x["difficulty"] for x in self.legal_actions(name)]:
            return False, "difficulty is not unlocked"
        return True, "valid"

    def fallback(self, benchmark: str) -> dict[str, Any]:
        return {"benchmark": benchmark, "difficulty": self.progress[benchmark].frontier}

    def record(self, benchmark: str, difficulty: int, outcome: dict[str, Any]) -> None:
        p = self.progress[benchmark]; fraction = float(outcome.get("fraction", 0.0)); p.attempts += 1
        p.first_success += bool(outcome.get("first_success")); p.final_success += bool(outcome.get("success"))
        p.partial_total += fraction; p.compile_failures += not bool(outcome.get("compiled", False)); p.failures += not bool(outcome.get("success"))
        p.recent = (p.recent + [fraction])[-5:]; trials = self.by_level[benchmark][difficulty]
        trials.append(bool(outcome.get("success"))); trials[:] = trials[-self.mastery_attempts:]
        if len(trials) == self.mastery_attempts and sum(trials) / len(trials) >= self.mastery_rate:
            if difficulty not in p.mastered: p.mastered.append(difficulty)
            nxt = min(self.max_difficulty, difficulty + 1)
            if nxt not in p.unlocked: p.unlocked.append(nxt)
            p.frontier = max(p.frontier, nxt)


def challenger_reward(action_valid: bool, used_fallback: bool, outcome: dict[str, Any], frontier_distance: int = 0) -> dict[str, float]:
    if not action_valid: return {"validity": -1.0, "frontier": 0.0, "useful": 0.0, "total": -1.0}
    if used_fallback: return {"validity": 0.0, "frontier": 0.0, "useful": 0.0, "total": 0.0}
    f = float(outcome.get("fraction", 0)); repair = bool(outcome.get("repair_success"))
    useful = 1.0 - min(1.0, abs(f - .6) / .6) + (.25 if repair else 0.0)
    frontier = max(0.0, 1.0 - .5 * abs(frontier_distance))
    return {"validity": .2, "frontier": frontier, "useful": useful, "total": round(.2 + frontier + useful, 4)}


@dataclass
class SolverResponse:
    strategy: str; reference_ids: list[str]; reference_usage: dict[str, str]; code: str; malformed: str | None = None


def parse_solver_contract(text: str) -> SolverResponse:
    import re
    clean = re.sub(r"```(?:[A-Za-z0-9_-]+)?\s*", "", text).replace("```", "").strip()
    def section(name: str) -> str | None:
        m = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", clean, re.S | re.I)
        return m.group(1).strip() if m else None
    strategy, refs, code = section("STRATEGY"), section("REFERENCE_USAGE"), section("CODE")
    if strategy is None or refs is None or not code: return SolverResponse(strategy or "", [], {}, code or "", "missing required contract section")
    if refs.upper() == "NONE": return SolverResponse(strategy, [], {}, code)
    usage = {}; malformed = None
    for line in refs.splitlines():
        m = re.match(r"\s*(R\d+)\s*-\s*(.+)", line, re.I)
        if not m: malformed = "invalid reference usage"; continue
        usage[m.group(1).upper()] = m.group(2).strip()
    return SolverResponse(strategy, list(usage), usage, code + "\n", malformed)


def reference_sections(text: str) -> dict[str, str]:
    """Give a supplied C reference stable IDs without changing its content."""
    chunks = [x.strip() for x in text.split("\n\n") if x.strip()]
    return {f"R{i}": chunk for i, chunk in enumerate(chunks, 1)}


def reference_identity(text: str, version: str = "unknown") -> dict[str, Any]:
    sections = reference_sections(text)
    return {"version": version, "sha256": hashlib.sha256(text.encode()).hexdigest(), "section_ids": list(sections)}


def normalize_resource_status(value: Any) -> dict[str, Any]:
    if value is None: return {"status": "unknown"}
    if isinstance(value, str): return {"status": value}
    if isinstance(value, dict): return {"status": str(value.get("status", "reported")), **value}
    return {"status": "unknown", "raw_type": type(value).__name__}


def _clip(value: Any, limit: int = 4000) -> dict[str, Any]:
    text = str(value or ""); return {"text": text[:limit], "truncated": len(text) > limit,
                                      "sha256": hashlib.sha256(text.encode()).hexdigest()}


def failure_feedback(task: Any, attempt: SolverResponse, result: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    build, correctness, perf = result.get("build", {}), result.get("correctness", {}), result.get("performance", {})
    passed, total = correctness.get("passed", 0), correctness.get("total", 0)
    return {"task": task, "contract": "STRATEGY/REFERENCE_USAGE/CODE", "reference": reference,
            "attempted_strategy": attempt.strategy, "previous_code": _clip(attempt.code),
            "compilation": {"success": build.get("exit_code") == 0, "diagnostics": _clip(build.get("stderr", ""))},
            "execution": {"exit_status": perf.get("exit_code"), "stderr": _clip(perf.get("stderr", "")), "timeout": perf.get("timed_out", False)},
            "tests": {"passed": passed, "failed": max(0, total - passed), "total": total, "cases": correctness.get("cases", [])[:10]},
            "resource": normalize_resource_status(perf.get("resource_status")), "reference_usage": attempt.reference_usage,
            "diagnosis": result.get("feedback", "objective judge failure")[:1000],
            "better_direction": "Fix compiler diagnostics first." if build.get("exit_code") not in (0, None) else "Address the first failing objective case."}


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...

class WhitespaceTokenizer:
    def count(self, text: str) -> int: return len(text.split())


def compose_context(parts: dict[str, str], config: ModelConfig, tokenizer: Tokenizer | None = None) -> tuple[str, dict[str, Any]]:
    tok = tokenizer or WhitespaceTokenizer(); reserved = int(config.generation.get("max_new_tokens", 2048)); budget = config.context_length - reserved
    order = ("system", "task", "challenge", "reference", "current", "feedback", "history", "memory")
    kept, dropped = [], []
    for name in order:
        value = parts.get(name, "")
        if tok.count("\n".join(kept + [value])) <= budget: kept.append(value)
        else: dropped.append(name)
    text = "\n".join(x for x in kept if x)
    return text, {"total_prompt_tokens": tok.count(text), "reference_tokens": tok.count(parts.get("reference", "")),
                  "feedback_tokens": tok.count(parts.get("feedback", "")), "history_tokens": tok.count(parts.get("history", "")),
                  "reserved_output_tokens": reserved, "truncated": bool(dropped), "removed": dropped}


def outcome_key(result: dict[str, Any]) -> tuple:
    parsed = bool(result.get("parsed", True)); compiled = bool(result.get("compiled", result.get("build", {}).get("exit_code") == 0))
    passed, total = int(result.get("passed", result.get("correctness", {}).get("passed", 0))), int(result.get("total", result.get("correctness", {}).get("total", 0)))
    full = total > 0 and passed == total
    return parsed, compiled, passed / max(1, total), full, -float(result.get("resource_metric", 0)) if full else 0


def verified_correction(attempts: list[tuple[SolverResponse, dict[str, Any], dict[str, Any]]]) -> dict[str, Any] | None:
    if len(attempts) < 2: return None
    best = max(range(len(attempts)), key=lambda i: outcome_key(attempts[i][1]))
    for i in range(best):
        if outcome_key(attempts[best][1]) > outcome_key(attempts[i][1]):
            old, old_result, feedback = attempts[i]; new, new_result, _ = attempts[best]
            return {"input": {"strategy": old.strategy, "code": old.code, "feedback": feedback},
                    "target": {"strategy": new.strategy, "code": new.code}, "before": outcome_key(old_result), "after": outcome_key(new_result)}
    return None


class Trainer(Protocol):
    def generate(self, role: str, prompt: str, config: dict[str, Any]) -> Any: ...
    def update_challenger(self, experience: dict[str, Any]) -> Any: ...
    def update_solver(self, examples: list[dict[str, Any]]) -> Any: ...
    def save_checkpoint(self, path: Path) -> Any: ...
    def load_checkpoint(self, path: Path) -> Any: ...
    def adapter_state(self, role: str) -> Any: ...
    def health(self) -> dict[str, Any]: ...


class MockTrainer:
    def __init__(self, replies: list[str] | None = None): self.replies = replies or []; self.updates: list[tuple[str, Any]] = []
    def generate(self, role, prompt, config): return self.replies.pop(0) if self.replies else ""
    def update_challenger(self, experience): self.updates.append(("challenger", experience)); return {"updated": True}
    def update_solver(self, examples): self.updates.append(("solver", examples)); return {"updated": bool(examples)}
    def save_checkpoint(self, path): Path(path).write_text("mock")
    def load_checkpoint(self, path): return Path(path).read_text()
    def adapter_state(self, role): return {"role": role, "mock": True}
    def health(self): return {"runtime": "mock", "healthy": True}


class HFPEFTTrainer:
    """Lazy hook for tomorrow's HF/PEFT implementation; never downloads on construction."""
    def __init__(self, model: ModelConfig): self.model = model
    def _unavailable(self):
        raise RuntimeError("HF/PEFT runtime is not loaded; install runtime dependencies and obtain the model on the GPU machine")
    generate = update_challenger = update_solver = save_checkpoint = load_checkpoint = adapter_state = _unavailable
    def health(self): return {"runtime": "hf-peft", "loaded": False, "model_id": self.model.model_id}


class EpisodeOrchestrator:
    """One idempotent episode transition; callers supply the benchmark judge."""
    def __init__(self, trainer: Trainer, curriculum: Curriculum, checkpoints: "Checkpoints", *, evaluation: bool = False, update_every: int = 1):
        self.trainer, self.curriculum, self.checkpoints = trainer, curriculum, checkpoints
        self.evaluation, self.update_every = evaluation, update_every

    def commit(self, run: dict[str, Any], episode_id: int, benchmark: str, action: dict[str, Any],
               attempts: list[tuple[SolverResponse, dict[str, Any], dict[str, Any]],], challenger_experience: dict[str, Any]) -> dict[str, Any]:
        if episode_id in run.get("committed_episodes", []): return {"duplicate": True, "updated": False}
        final = attempts[-1][1]; fraction = final.get("passed", final.get("correctness", {}).get("passed", 0)) / max(1, final.get("total", final.get("correctness", {}).get("total", 0)))
        outcome = {"fraction": fraction, "success": fraction == 1, "compiled": outcome_key(final)[1],
                   "first_success": bool(attempts and outcome_key(attempts[0][1])[3]), "repair_success": len(attempts) > 1 and fraction == 1 and not outcome_key(attempts[0][1])[3]}
        valid, _ = self.curriculum.validate(action)
        reward = challenger_reward(valid, bool(challenger_experience.get("used_fallback")), outcome,
                                   action.get("difficulty", 1) - self.curriculum.progress[benchmark].frontier)
        correction = verified_correction(attempts)
        updated = False
        if not self.evaluation and episode_id % self.update_every == 0:
            self.curriculum.record(benchmark, action["difficulty"], outcome)
            self.trainer.update_challenger({**challenger_experience, "reward": reward, "episode_id": episode_id})
            if correction: self.trainer.update_solver([correction])
            updated = True
        run.setdefault("episodes", []).append({"episode_id": episode_id, "action": action, "outcome": outcome,
                                                "challenger_reward": reward, "verified_correction": bool(correction), "updated": updated})
        self.checkpoints.commit_episode(run, episode_id)
        return {"duplicate": False, "updated": updated, "reward": reward, "correction": correction, "outcome": outcome}


class Checkpoints:
    def __init__(self, root: Path, milestones: int = 3): self.root, self.milestones = Path(root), milestones
    def save(self, state: dict[str, Any], name: str = "latest") -> Path:
        target = self.root / name; target.mkdir(parents=True, exist_ok=True); temp = target / "state.json.tmp"; final = target / "state.json"
        temp.write_text(json.dumps(state, indent=2, default=str)); os.replace(temp, final); return final
    def commit_episode(self, state: dict[str, Any], episode_id: int) -> Path:
        state.setdefault("committed_episodes", [])
        if episode_id not in state["committed_episodes"]: state["committed_episodes"].append(episode_id)
        self.save(state); path = self.save(state, f"milestone-{episode_id:06d}")
        old = sorted(self.root.glob("milestone-*"))[:-self.milestones]
        for item in old: shutil.rmtree(item)
        return path
    def load_latest(self) -> dict[str, Any]: return json.loads((self.root / "latest/state.json").read_text())


def manifest(model: ModelConfig, train_manifest: Any, eval_manifest: Any, **extra: Any) -> dict[str, Any]:
    def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()
    try: commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip()
    except OSError: commit = None
    return {"git_commit": commit, "dirty": bool(subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True).stdout.strip()),
            "model": asdict(model), "train_benchmark_hash": digest(train_manifest), "eval_benchmark_hash": digest(eval_manifest),
            "doctor": hardware_profile(), **extra}


def validate_manifests(train: list[Any], heldout: list[Any]) -> None:
    left, right = {json.dumps(x, sort_keys=True) for x in train}, {json.dumps(x, sort_keys=True) for x in heldout}
    if left & right: raise ValueError("train/eval benchmark manifests overlap")


def select_judge(backend: str):
    if backend == "docker":
        from .sandbox import run_c
        return run_c
    if backend == "wsl": return WslJudge()
    if backend == "windows-native": return WindowsNativeJudge()
    raise ValueError("unknown judge backend")


class WslJudge:
    backend = "wsl"
    def run(self, *args, **kwargs): raise RuntimeError("WSL judge is unverified; run doctor/smoke test on Windows")


class WindowsNativeJudge:
    backend = "windows-native"
    def run(self, *args, **kwargs):
        if os.name != "nt": raise RuntimeError("Windows-native judge is unverified outside Windows")
        raise RuntimeError("Windows Job Object runner needs college-PC smoke validation")


def estimate_storage(episodes: int, average_evidence_bytes: int, adapter_bytes: int = 0, milestone_every: int = 10) -> int:
    return episodes * average_evidence_bytes + ((episodes + milestone_every - 1) // milestone_every) * adapter_bytes
