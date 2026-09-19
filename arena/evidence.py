"""Versioned episode evidence and viewer events, independent of benchmark logic."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1
REQUIRED = (
    "run_id", "episode_id", "benchmark", "seeds", "git_before", "challenger",
    "solver", "input_generation", "build", "correctness", "performance",
    "judge", "rewards", "outcome", "git_after", "timestamps",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def viewer_events(record: dict) -> list[dict]:
    """Chronological actor cards consumed by the future viewer."""
    base = {"run_id": record["run_id"], "episode_id": record["episode_id"]}
    return [
        {**base, "schema_version": SCHEMA_VERSION, "actor": "challenger", "kind": "challenge",
         "at": record["timestamps"]["started_at"],
         "request": record["challenger"]["request"],
         "rationale": record["challenger"].get("rationale")},
        {**base, "schema_version": SCHEMA_VERSION, "actor": "solver", "kind": "candidate",
         "at": record["timestamps"]["candidate_at"],
         "response": record["solver"]["response"],
         "candidate": record["solver"]["candidate"],
         "patch": record["solver"].get("patch")},
        {**base, "schema_version": SCHEMA_VERSION, "actor": "judge", "kind": "verdict",
         "at": record["timestamps"]["finished_at"],
         "correctness": record["correctness"],
         "performance": record["performance"],
         "resource_usage": record["judge"].get("resource_usage", {}),
         "feedback": record["judge"]["feedback"],
         "rewards": record["rewards"], "outcome": record["outcome"]},
    ]


def write_episode(root: str | Path, record: dict) -> tuple[Path, Path]:
    """Retain one complete episode as JSON, Markdown, and append-only viewer events."""
    missing = set(REQUIRED) - record.keys()
    if missing:
        raise ValueError(f"episode missing: {', '.join(sorted(missing))}")
    run_id, episode_id = record["run_id"], record["episode_id"]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("run_id must be a short path-safe identifier")
    if not isinstance(episode_id, int) or isinstance(episode_id, bool) or episode_id < 1:
        raise ValueError("episode_id must be a positive integer")
    if record["outcome"] not in ("accepted", "rejected"):
        raise ValueError("outcome must be accepted or rejected")
    if record["outcome"] == "accepted" and not record["git_after"]:
        raise ValueError("accepted episode needs resulting Git commit")
    if record["outcome"] == "rejected" and record["git_after"]:
        raise ValueError("rejected episode cannot have resulting Git commit")
    if not isinstance(record["seeds"], dict) or not isinstance(record["input_generation"], dict):
        raise ValueError("seeds and input_generation must be objects")
    if not isinstance(record["timestamps"], dict):
        raise ValueError("timestamps must be an object")
    times = [datetime.fromisoformat(record["timestamps"][key])
             for key in ("started_at", "candidate_at", "finished_at")]
    if not times[0] <= times[1] <= times[2]:
        raise ValueError("episode timestamps are out of order")
    for key in ("solver", "challenger", "judge", "rewards", "build",
                "correctness", "performance"):
        if not isinstance(record[key], dict):
            raise ValueError(f"{key} must be an object")
    for key in ("solver", "challenger"):
        if ("candidate" if key == "solver" else "request") not in record[key]:
            raise ValueError(f"{key} content missing")
    if "feedback" not in record["judge"] or not {"solver", "challenger"} <= record["rewards"].keys():
        raise ValueError("judge feedback and both rewards required")
    if any(not isinstance(record["rewards"][key], (int, float)) or
           not math.isfinite(record["rewards"][key]) for key in ("solver", "challenger")):
        raise ValueError("rewards must be finite numbers")
    if not {"generator", "seed", "config"} <= record["input_generation"].keys():
        raise ValueError("input generation recipe incomplete")
    if not {"exit_code", "stdout", "stderr"} <= record["build"].keys():
        raise ValueError("build must retain compiler exit code, stdout and stderr")

    saved = dict(record)
    saved["schema_version"] = SCHEMA_VERSION
    saved["hashes"] = {
        "candidate_sha256": sha256_text(record["solver"]["candidate"]),
        "challenge_sha256": sha256_text(json.dumps(record["challenger"]["request"], sort_keys=True)),
    }
    if record["solver"].get("patch") is not None:
        saved["hashes"]["patch_sha256"] = sha256_text(record["solver"]["patch"])
    events = viewer_events(saved)
    directory = Path(root) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{episode_id:06d}"
    json_path, md_path = directory / f"{stem}.json", directory / f"{stem}.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError(f"episode {episode_id} already exists in run {run_id}")
    json_path.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(
        f"# Episode {episode_id}: {record['benchmark']}\n\n"
        f"- Outcome: {record['outcome']}\n"
        f"- Git: {record['git_before']} → {record['git_after'] or 'unchanged'}\n"
        f"- Solver reward: {record['rewards']['solver']}\n"
        f"- Challenger reward: {record['rewards']['challenger']}\n\n"
        f"## Challenger\n\n{record['challenger']['request']}\n\n"
        f"Rationale: {record['challenger'].get('rationale') or 'not supplied'}\n\n"
        f"## Solver\n\n{record['solver']['response']}\n\n"
        f"## Judge\n\n{record['judge']['feedback']}\n",
        encoding="utf-8",
    )
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    return json_path, md_path
