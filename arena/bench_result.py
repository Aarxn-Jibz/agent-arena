"""Shared evidence handoff after a benchmark Judge has finished."""

from __future__ import annotations

from .evidence import write_episode


def record_judgement(root, *, benchmark: str, run_id: str, episode_id: int,
                     challenge: dict, source: str, evaluation: dict,
                     git_before: str, git_after: str | None,
                     solver_response: str = "", rationale: str | None = None,
                     patch: str | None = None, reference_reads: list[dict] | None = None):
    if evaluation["accepted"] != bool(git_after):
        raise ValueError("git_after must be supplied exactly when Judge accepts")
    seed = challenge["seed"]
    record = {"run_id": run_id, "episode_id": episode_id, "benchmark": benchmark,
              "seeds": {"corpus": seed}, "git_before": git_before,
              "challenger": {"request": challenge, "rationale": rationale},
              "solver": {"response": solver_response, "candidate": source, "patch": patch},
              "reference_reads": reference_reads or [],
              "input_generation": {"generator": f"{benchmark}-v1", "seed": seed,
                                   "config": challenge},
              "build": evaluation["build"], "correctness": evaluation["correctness"],
              "performance": evaluation["performance"],
              "judge": {"feedback": evaluation["feedback"],
                        "reward_inputs": evaluation["reward_inputs"],
                        "resource_usage": evaluation.get("resource_usage", {"memory_bytes": None})},
              "rewards": {"solver": evaluation["reward_inputs"]["solver_reward"],
                          "challenger": evaluation["reward_inputs"]["challenger_reward"]},
              "outcome": "accepted" if evaluation["accepted"] else "rejected",
              "git_after": git_after,
              "timestamps": {"started_at": evaluation["started_at"],
                             "candidate_at": evaluation["started_at"],
                             "finished_at": evaluation["finished_at"]}}
    return write_episode(root, record)
