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
            "windows_judge": "WINDOWS JUDGE: UNVERIFIED — SMOKE TEST REQUIRED"}


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
    """HF/PEFT LoRA runtime.  Construction and tests never touch the network.

    ``load()`` is intentionally explicit: tomorrow's run configuration decides
    whether weights may be fetched.  Two named adapters share frozen base
    weights while their optimizers and counters remain independent.
    """
    ROLES = ("challenger", "solver")

    def __init__(self, model: ModelConfig, *, allow_download: bool = False, dependencies=None):
        self.model_config, self.allow_download, self._injected = model, allow_download, dependencies
        self.model = self.tokenizer = None
        self.optimizers: dict[str, Any] = {}; self.steps = {role: 0 for role in self.ROLES}
        self.baseline = 0.0; self.active_adapter = None

    def _dependencies(self):
        if self._injected: return self._injected
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            return {"torch": torch, "AutoModelForCausalLM": AutoModelForCausalLM,
                    "AutoTokenizer": AutoTokenizer, "BitsAndBytesConfig": BitsAndBytesConfig,
                    "LoraConfig": LoraConfig, "TaskType": TaskType, "get_peft_model": get_peft_model,
                    "prepare_model_for_kbit_training": prepare_model_for_kbit_training}
        except ImportError as err:
            raise RuntimeError("HF/PEFT runtime requires torch, transformers and peft (and bitsandbytes for 4-bit QLoRA)") from err

    def _dtype(self, torch):
        names = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
        try: return getattr(torch, names[self.model_config.precision])
        except KeyError as err: raise ValueError("precision must be bf16, fp16, or fp32") from err

    def load(self):
        if self.model is not None: return self
        d, torch = self._dependencies(), None
        torch = d["torch"]; local = not self.allow_download
        tokenizer_revision = self.model_config.tokenizer_revision or self.model_config.revision
        self.tokenizer = d["AutoTokenizer"].from_pretrained(self.model_config.model_id, revision=tokenizer_revision,
                                                               local_files_only=local)
        kwargs = {"revision": self.model_config.revision, "local_files_only": local,
                  "torch_dtype": self._dtype(torch)}
        if self.model_config.quantization == "4bit":
            kwargs["quantization_config"] = d["BitsAndBytesConfig"](load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self._dtype(torch), bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = "auto"
        elif self.model_config.quantization != "none":
            raise ValueError("quantization must be none or 4bit")
        self.model = d["AutoModelForCausalLM"].from_pretrained(self.model_config.model_id, **kwargs)
        if self.model_config.quantization == "4bit": self.model = d["prepare_model_for_kbit_training"](self.model)
        lora = self.model_config.lora
        targets = lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        config = d["LoraConfig"](r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", .05)), target_modules=targets, bias="none",
            task_type=d["TaskType"].CAUSAL_LM)
        self.model = d["get_peft_model"](self.model, config, adapter_name="challenger")
        self.model.add_adapter("solver", config)
        for parameter in self.model.parameters(): parameter.requires_grad = False
        for role in self.ROLES:
            self.set_adapter(role)
            params = self._role_parameters(role)
            if not params: raise RuntimeError(f"{role} adapter has zero trainable parameters")
            lr = float(lora.get(f"{role}_lr", lora.get("learning_rate", 2e-4)))
            self.optimizers[role] = torch.optim.AdamW(params, lr=lr)
        self.set_adapter("challenger")
        self.validate_trainable_parameters()
        return self

    def _role_parameters(self, role):
        # PEFT parameter names contain their named adapter; this also keeps
        # optimizer state independent even though adapters share a backbone.
        return [p for name, p in self.model.named_parameters() if role in name and getattr(p, "requires_grad", False)]

    def set_adapter(self, role: str):
        if role not in self.ROLES: raise ValueError("role must be challenger or solver")
        if self.model is None: self.load()
        self.model.set_adapter(role)
        for name, parameter in self.model.named_parameters(): parameter.requires_grad = role in name
        self.active_adapter = role
        return role

    def validate_trainable_parameters(self):
        if self.model is None: self.load()
        total = sum(p.numel() for p in self.model.parameters())
        report = {"total_parameters": total, "active_adapter": self.active_adapter}
        for role in self.ROLES:
            self.set_adapter(role); count = sum(p.numel() for p in self._role_parameters(role))
            if not count: raise RuntimeError(f"{role} adapter has zero trainable parameters")
            report[f"{role}_trainable_parameters"] = count
            report[f"{role}_percent_trainable"] = 100 * count / max(1, total)
        base_trainable = [name for name, p in self.model.named_parameters()
                          if p.requires_grad and not any(role in name for role in self.ROLES)]
        if base_trainable: raise RuntimeError("base model parameters are trainable: " + ", ".join(base_trainable[:3]))
        self.set_adapter(report["active_adapter"])
        return report

    def _encode(self, text: str):
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in encoded.items()}

    def count(self, text: str) -> int:
        """Tokenizer-derived count for ``compose_context(..., trainer)``."""
        self.load()
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def compose_prompt(self, parts: dict[str, str]):
        """Apply the shared priority policy using this model's tokenizer."""
        return compose_context(parts, self.model_config, self)

    def generate(self, role: str, prompt: str, config: dict[str, Any]):
        self.load(); self.set_adapter(role)
        d = self._dependencies(); torch = d["torch"]; inputs = self._encode(prompt)
        maximum = int(config.get("max_new_tokens", self.model_config.generation["max_new_tokens"]))
        if inputs["input_ids"].shape[1] + maximum > self.model_config.context_length: raise ValueError("prompt exceeds configured context budget")
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=maximum, do_sample=config.get("do_sample", True),
                temperature=float(config.get("temperature", self.model_config.generation.get("temperature", .7))),
                top_p=float(config.get("top_p", self.model_config.generation.get("top_p", .9))),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def action_log_probability(self, prompt: str, action: dict[str, Any]):
        """Differentiable log P(action JSON | prompt), retained for REINFORCE."""
        self.load(); self.set_adapter("challenger"); torch = self._dependencies()["torch"]
        action_text = json.dumps(action, sort_keys=True, separators=(",", ":"))
        prefix, full = self._encode(prompt), self._encode(prompt + action_text)
        prompt_tokens = prefix["input_ids"].shape[1]; output = self.model(**full)
        logits, ids = output.logits[:, :-1, :], full["input_ids"][:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        return log_probs[:, max(0, prompt_tokens - 1):].sum()

    def sample_challenger_action(self, prompt: str, legal_actions: list[dict[str, Any]]):
        """Sample only from legal actions and retain its differentiable log-P."""
        if not legal_actions: raise ValueError("challenger has no legal actions")
        self.load(); torch = self._dependencies()["torch"]
        scores = torch.stack([self.action_log_probability(prompt, action) for action in legal_actions])
        distribution = torch.distributions.Categorical(logits=scores)
        choice = distribution.sample()
        index = int(choice.item())
        return {"action": legal_actions[index], "log_probability": distribution.log_prob(choice), "valid": True,
                "action_index": index, "legal_action_count": len(legal_actions)}

    def _solver_loss(self, example: dict[str, Any]):
        self.load(); torch = self._dependencies()["torch"]
        source, target = example["input"], example["target"]
        prompt = json.dumps(source, sort_keys=True) + "\n<REPAIR>\n"
        completion = ("<STRATEGY>\n" + target["strategy"] + "\n</STRATEGY>\n<CODE>\n" + target["code"] + "\n</CODE>")
        encoded, prompt_ids = self._encode(prompt + completion), self._encode(prompt)["input_ids"].shape[1]
        labels = encoded["input_ids"].clone(); labels[:, :prompt_ids] = -100
        return self.model(**encoded, labels=labels).loss

    def _step(self, role: str, loss):
        torch = self._dependencies()["torch"]; optimizer = self.optimizers[role]
        loss.backward(); torch.nn.utils.clip_grad_norm_(self._role_parameters(role), float(self.model_config.lora.get("max_grad_norm", 1.0)))
        optimizer.step(); optimizer.zero_grad(set_to_none=True); self.steps[role] += 1
        return float(loss.detach().cpu())

    def update_solver(self, examples: list[dict[str, Any]]):
        if not examples: return {"updated": False, "reason": "no verified corrections"}
        self.load(); self.set_adapter("solver"); losses = [self._solver_loss(example) for example in examples]
        loss = sum(losses) / len(losses); value = self._step("solver", loss)
        return {"updated": True, "loss": value, "examples": len(examples), "step": self.steps["solver"], "adapter": self.active_adapter}

    def update_challenger(self, experience: dict[str, Any]):
        if experience.get("used_fallback") or not experience.get("valid", True) or experience.get("log_probability") is None:
            return {"updated": False, "reason": "fallback, illegal action, or missing log probability"}
        self.load(); self.set_adapter("challenger"); reward = experience.get("reward", 0.0)
        reward = reward["total"] if isinstance(reward, dict) else float(reward)
        advantage = reward - self.baseline; self.baseline = .9 * self.baseline + .1 * reward
        value = self._step("challenger", -advantage * experience["log_probability"])
        return {"updated": True, "loss": value, "reward": reward, "advantage": advantage,
                "baseline": self.baseline, "step": self.steps["challenger"], "adapter": self.active_adapter}

    def save_checkpoint(self, path: Path):
        self.load(); path = Path(path); path.mkdir(parents=True, exist_ok=True); torch = self._dependencies()["torch"]
        for role in self.ROLES:
            self.model.save_pretrained(path / role, selected_adapters=[role])
            torch.save(self.optimizers[role].state_dict(), path / f"{role}-optimizer.pt")
        (path / "trainer.json").write_text(json.dumps({"steps": self.steps, "baseline": self.baseline, "model": asdict(self.model_config)}))
        return path

    def load_checkpoint(self, path: Path):
        self.load(); path = Path(path); torch = self._dependencies()["torch"]
        for role in self.ROLES:
            self.model.load_adapter(path / role, adapter_name=role, is_trainable=True)
            self.optimizers[role].load_state_dict(torch.load(path / f"{role}-optimizer.pt", map_location="cpu", weights_only=True))
        saved = json.loads((path / "trainer.json").read_text()); self.steps = saved["steps"]; self.baseline = saved["baseline"]
        self.set_adapter("challenger"); return self.health()

    def adapter_state(self, role: str):
        self.load(); self.set_adapter(role)
        return {"role": role, "step": self.steps[role], "trainable": sum(p.numel() for p in self._role_parameters(role))}

    def health(self):
        return {"runtime": "hf-peft", "loaded": self.model is not None, "model_id": self.model_config.model_id,
                "active_adapter": self.active_adapter, "steps": dict(self.steps), "allow_download": self.allow_download}


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
        if not self.evaluation:
            self.checkpoints.save_trainer(self.trainer, "latest")
            self.checkpoints.save_trainer(self.trainer, f"milestone-{episode_id:06d}")
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
    def save_trainer(self, trainer: Trainer, name: str = "latest") -> Path:
        target = self.root / name; target.mkdir(parents=True, exist_ok=True)
        return trainer.save_checkpoint(target / "trainer")
    def load_trainer(self, trainer: Trainer, name: str = "latest"):
        return trainer.load_checkpoint(self.root / name / "trainer")


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
