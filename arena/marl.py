"""Minimal adversarial MARL prototype with tabular Q-learning.

Two agents learn meta-policies over a **frozen** SmolLM2 backbone:

- Challenger agent: which existing task to present (task *selection*).
- Solver agent: which generic prompting strategy to use.

Both observe a shared skill state (low/medium/high) and are updated with
standard tabular Q-learning from the rewards produced by the existing
deterministic TCC arena. The arena judge remains authoritative; SmolLM2's
weights are never updated. This is independent Q-learning at the
agent/meta-policy level, not RL of the language model.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from .solver import PROMPT_STRATEGIES, solve

STATES = ("low", "medium", "high")
TASK_IDS = ("001", "002", "003")
STRATEGIES = tuple(PROMPT_STRATEGIES)  # plain, strict, few_shot

# Skill-state buckets (average of the last up to 3 best_pass_fraction values).
LOW_BOUND = 0.34
HIGH_BOUND = 0.67
HISTORY_WINDOW = 3

# Known full-success arena reward, used to normalize the Solver learning reward.
FULL_REWARD = 16.0

DEFAULT_ALPHA = 0.4
DEFAULT_GAMMA = 0.8
DEFAULT_EPSILON = 0.3
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Shared environment state
# ---------------------------------------------------------------------------

def rolling_average(history: list[float]) -> float:
    """Mean of the last up to HISTORY_WINDOW values; 0.0 when empty."""
    recent = history[-HISTORY_WINDOW:]
    return sum(recent) / len(recent) if recent else 0.0


def skill_state(avg: float) -> str:
    """Bucket a recent-average best-pass-fraction into a skill state."""
    if avg < LOW_BOUND:
        return "low"
    if avg < HIGH_BOUND:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------

def init_q(actions) -> dict:
    """All Q values start at 0.0: {state: {action: 0.0, ...}, ...}."""
    return {state: {action: 0.0 for action in actions} for state in STATES}


def q_update(q, state: str, action: str, reward: float, next_state: str,
             alpha: float, gamma: float) -> float:
    """Bellman-style tabular update: Q(s,a) += alpha*(r + gamma*max Q(s',.) - Q(s,a))."""
    old = q[state][action]
    target = reward + gamma * max(q[next_state].values())
    new = round(old + alpha * (target - old), 6)
    q[state][action] = new
    return new


def choose_action(q_row: dict, epsilon: float, rng: random.Random) -> str:
    """Epsilon-greedy action selection; seeded RNG breaks ties randomly."""
    if rng.random() < epsilon:
        return rng.choice(list(q_row))
    best = max(q_row.values())
    return rng.choice([a for a, v in q_row.items() if v == best])


def greedy_action(q_row: dict, rng: random.Random) -> str:
    """Greedy action (epsilon=0) used for final policy reports."""
    return choose_action(q_row, 0.0, rng)


# ---------------------------------------------------------------------------
# Learning rewards (derived from the existing deterministic judge only)
# ---------------------------------------------------------------------------

def solver_reward(existing_rewards: list[float]) -> float:
    """best_existing_reward / FULL_REWARD, clamped to [-1.0, 1.0]."""
    best = max(existing_rewards) if existing_rewards else 0.0
    scaled = best / FULL_REWARD
    return max(-1.0, min(1.0, scaled))


def challenger_reward(mean_pass_fraction: float) -> float:
    """Frontier reward: highest when the task sits near 0.5 pass fraction.

    mean=0.5 -> 1.0 (frontier), mean=0.0 or 1.0 -> 0.0 (useless/trivial).
    """
    return max(0.0, 1.0 - 2.0 * abs(mean_pass_fraction - 0.5))


# ---------------------------------------------------------------------------
# Episode outcome helpers
# ---------------------------------------------------------------------------

def _summarize(result) -> dict:
    """Summarize one Solver run for MARL (arena evidence is authoritative)."""
    per_attempt = [(r["passed"], r["total"]) for r in result.eval_results]
    pass_fractions = [
        (r["passed"] / r["total"]) if r["compiled"] and r["total"] else 0.0
        for r in result.eval_results
    ]
    best_idx = max(range(len(pass_fractions)), key=lambda i: pass_fractions[i]) \
        if pass_fractions else None
    return {
        "success": result.success,
        "attempts": result.attempts,
        "per_attempt": per_attempt,
        "best_tests": per_attempt[best_idx] if best_idx is not None else (0, 0),
        "pass_fractions": pass_fractions,
        "best_pass_fraction": pass_fractions[best_idx] if best_idx is not None else 0.0,
        "mean_pass_fraction": sum(pass_fractions) / len(pass_fractions) if pass_fractions else 0.0,
        "arena_rewards": list(result.rewards),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def default_state(alpha: float, gamma: float, epsilon: float, seed: int) -> dict:
    """A fresh MARL state: zero Q tables, empty history, no episodes done."""
    return {
        "Q_challenger": init_q(TASK_IDS),
        "Q_solver": init_q(STRATEGIES),
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "seed": seed,
        "episodes_completed": 0,
        "history": [],
        "rng_state": None,
    }


def load_state(path, alpha: float, gamma: float, epsilon: float, seed: int) -> dict:
    """Load a run's exact continuation state."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if any(raw.get(k) != v for k, v in
           {"alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}.items()):
        raise ValueError("resume requires the original alpha, gamma, epsilon and seed")
    if raw.get("episodes_completed", 0) and "rng_state" not in raw:
        raise ValueError("saved state lacks policy RNG state; exact resume is impossible")
    qc = {s: {a: 0.0 for a in TASK_IDS} for s in STATES}
    qc.update({s: {a: raw["Q_challenger"].get(s, {}).get(a, 0.0) for a in TASK_IDS}
               for s in STATES})
    qs = {s: {a: 0.0 for a in STRATEGIES} for s in STATES}
    qs.update({s: {a: raw["Q_solver"].get(s, {}).get(a, 0.0) for a in STRATEGIES}
               for s in STATES})
    return {
        "Q_challenger": qc,
        "Q_solver": qs,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "seed": seed,
        "episodes_completed": int(raw.get("episodes_completed", 0)),
        "history": [float(x) for x in raw.get("history", [])][-HISTORY_WINDOW:],
        "rng_state": raw.get("rng_state"),
    }


def save_state(path, state: dict) -> None:
    """Persist after every episode so an interrupted run can resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Episode log
# ---------------------------------------------------------------------------

def append_episode_log(path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _seed_torch(value: int) -> None:
    """Seed torch's RNG before each episode's model sampling.

    Deliberately a no-op when torch is absent (unit tests). Exact
    floating-point/model determinism across different chips/software stacks is
    not guaranteed; this gives a stable random sequence "where practical".
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    torch.manual_seed(value)


def run_episodes(tasks, generate, *, episodes: int = 9, attempts: int = 2,
                 alpha: float = DEFAULT_ALPHA, gamma: float = DEFAULT_GAMMA,
                 epsilon: float = DEFAULT_EPSILON, seed: int = DEFAULT_SEED,
                 state_path=None, log_path=None, on_episode=None):
    """Run `episodes` MARL episodes and return (records, state).

    `tasks` maps task id -> task dict; `generate(messages)` drives the frozen
    model (loaded once by the caller). Q tables persist to `state_path` and
    episode records append to `log_path` after every episode.
    """
    if isinstance(tasks, (list, tuple)):
        tasks = {t["id"]: t for t in tasks}

    if state_path is not None and Path(state_path).exists():
        state = load_state(state_path, alpha, gamma, epsilon, seed)
    else:
        state = default_state(alpha, gamma, epsilon, seed)
    state.update(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)

    rng = random.Random(seed)
    if state["rng_state"] is not None:
        rng.setstate((state["rng_state"][0], tuple(state["rng_state"][1]),
                      state["rng_state"][2]))
    records = []
    for _ in range(episodes):
        episode = state["episodes_completed"] + 1
        cur_state = skill_state(rolling_average(state["history"]))
        task_id = choose_action(state["Q_challenger"][cur_state], epsilon, rng)
        strategy = choose_action(state["Q_solver"][cur_state], epsilon, rng)

        _seed_torch(seed + episode)  # stable per-episode sampling sequence
        result = solve(
            tasks[task_id], generate, max_attempts=attempts,
            prompt_builder=PROMPT_STRATEGIES[strategy],
        )
        outcome = _summarize(result)

        solver_rl = solver_reward(result.rewards)
        challenger_rl = challenger_reward(outcome["mean_pass_fraction"])

        state["history"].append(outcome["best_pass_fraction"])
        state["history"] = state["history"][-HISTORY_WINDOW:]
        next_state = skill_state(rolling_average(state["history"]))

        q_update(state["Q_challenger"], cur_state, task_id, challenger_rl,
                 next_state, alpha, gamma)
        q_update(state["Q_solver"], cur_state, strategy, solver_rl,
                 next_state, alpha, gamma)
        state["episodes_completed"] += 1
        state["rng_state"] = rng.getstate()

        record = {
            "episode": episode,
            "state": cur_state,
            "challenger_action": task_id,
            "solver_action": strategy,
            "success": outcome["success"],
            "attempts": outcome["attempts"],
            "best_tests": outcome["best_tests"],
            "best_pass_fraction": outcome["best_pass_fraction"],
            "mean_pass_fraction": outcome["mean_pass_fraction"],
            "arena_rewards": outcome["arena_rewards"],
            "solver_reward": solver_rl,
            "challenger_reward": challenger_rl,
            "next_state": next_state,
            "episode_seed": seed + episode,
            "Q_challenger_after": dict(state["Q_challenger"][cur_state]),
            "Q_solver_after": dict(state["Q_solver"][cur_state]),
        }
        records.append(record)
        if log_path:
            append_episode_log(log_path, record)
        if state_path:
            save_state(state_path, state)
        if on_episode:
            on_episode(record)

    return records, state


def summarize_run(records: list[dict], state: dict) -> dict:
    """Summarize only episodes executed in this invocation."""
    count = len(records)
    average = lambda key: sum(r[key] for r in records) / count if count else 0.0
    return {
        "episodes": count,
        "start_episode": records[0]["episode"] if records else state["episodes_completed"] + 1,
        "end_episode": state["episodes_completed"],
        "average_tests_passed": average("best_pass_fraction"),
        "average_solver_reward": average("solver_reward"),
        "average_challenger_reward": average("challenger_reward"),
        "tasks": Counter(r["challenger_action"] for r in records),
        "prompts": Counter(r["solver_action"] for r in records),
        "policy": {s: {
            "task": max(state["Q_challenger"][s], key=state["Q_challenger"][s].get),
            "prompt": max(state["Q_solver"][s], key=state["Q_solver"][s].get),
        } for s in STATES},
        "failures": sum(not r["success"] for r in records),
    }


def format_report(summary: dict, state: dict, attempts: int, state_path, log_path,
                  report_path=None) -> str:
    """Human-readable report for one invocation; counts exclude earlier resumed episodes."""
    lines = [
        "# MARL run report", "",
        "Policy selection over a frozen model; no model weight updates or measured improvement.", "",
        "## Configuration", "",
        f"- Seed: {state['seed']}",
        f"- Episodes this run: {summary['episodes']}",
        f"- Start/end episode: {summary['start_episode']}–{summary['end_episode']}",
        f"- Maximum attempts per episode: {attempts}",
        f"- Alpha / gamma / epsilon: {state['alpha']} / {state['gamma']} / {state['epsilon']}",
        "", "## Results", "",
        f"- Average best tests passed: {summary['average_tests_passed']:.1%}",
        f"- Average Solver reward: {summary['average_solver_reward']:.3f}",
        f"- Average Challenger reward: {summary['average_challenger_reward']:.3f}",
        f"- Episodes without full success: {summary['failures']}",
        "", "## Task selection", "",
    ]
    lines += [f"- {a}: {summary['tasks'][a]}" for a in TASK_IDS]
    lines += ["", "## Prompt selection", ""]
    lines += [f"- {a}: {summary['prompts'][a]}" for a in STRATEGIES]
    lines += ["", "## Final learned policy by state", ""]
    lines += [f"- {s}: task {summary['policy'][s]['task']}, prompt {summary['policy'][s]['prompt']}"
              for s in STATES]
    lines += ["", "## Artifacts", "", f"- Q state: {state_path}",
              f"- Episode JSONL: {log_path}"]
    if report_path is not None:
        lines.append(f"- Report: {report_path}")
    lines.append("")
    return "\n".join(lines)
