"""Minimal JSONL trajectory logging: one JSON object per line per attempt."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def trajectory_entry(task: dict, attempt: int, source: str, result: dict) -> dict:
    """Build one trajectory record.

    `result` is the full evaluation result as a dict (e.g.
    dataclasses.asdict(evaluate(...))); reward and success are copied to the
    top level for quick reading.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": task["id"],
        "attempt": attempt,
        "source": source,
        "result": result,
        "reward": result["reward"],
        "success": result["success"],
    }


def append_trajectory(path: str | os.PathLike, entry: dict) -> None:
    """Append one JSON object on its own line, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")