"""Read-only localhost viewer for generic episode evidence."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

PAGE = (Path(__file__).parent.parent / "viewer" / "index.html")
ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")


def episodes(root: Path, run_id: str) -> list[dict]:
    if not ID.fullmatch(run_id):
        raise ValueError("invalid run ID")
    found = []
    for path in sorted((root / run_id).glob("[0-9]*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("run_id") == run_id and isinstance(data.get("episode_id"), int):
                found.append(data)
        except (OSError, ValueError, UnicodeError):
            continue  # live writer may not have finished this file
    return sorted(found, key=lambda x: x["episode_id"])


def episode_detail(records: list[dict], episode_id: int) -> dict | None:
    prior = None
    for record in records:
        if record["episode_id"] == episode_id:
            candidate = record.get("solver", {}).get("candidate", "")
            record = dict(record)
            record["source_diff"] = record.get("solver", {}).get("patch") or ("".join(difflib.unified_diff(
                prior.splitlines(keepends=True), candidate.splitlines(keepends=True),
                fromfile="previous accepted", tofile="candidate")) if prior is not None
                else "No previous accepted version in this run.")
            return record
        if record.get("outcome") == "accepted":
            prior = record.get("solver", {}).get("candidate", "")
    return None


def serve(root: Path, port: int = 8765) -> ThreadingHTTPServer:
    root = root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = [unquote(x) for x in urlsplit(self.path).path.strip("/").split("/")]
            try:
                if parts == [""]:
                    body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
                elif parts == ["api", "runs"]:
                    body, kind = json.dumps(sorted(p.name for p in root.iterdir()
                                                   if p.is_dir() and ID.fullmatch(p.name))).encode(), "application/json"
                elif len(parts) >= 3 and parts[:2] == ["api", "runs"] and ID.fullmatch(parts[2]):
                    run_id = parts[2]
                    records = episodes(root, run_id)
                    if len(parts) == 4 and parts[3] == "episodes":
                        rows = [{"episode_id": r["episode_id"], "benchmark": r.get("benchmark"),
                                 "challenge": str(r.get("challenger", {}).get("request", ""))[:120],
                                 "outcome": r.get("outcome"),
                                 "headline": r.get("judge", {}).get("feedback", "")}
                                for r in records]
                        body, kind = json.dumps(rows).encode(), "application/json"
                    elif len(parts) == 5 and parts[3] == "episodes" and parts[4].isdigit():
                        record = episode_detail(records, int(parts[4]))
                        if record is None:
                            self.send_error(404)
                            return
                        body, kind = json.dumps(record).encode(), "application/json"
                    elif len(parts) == 4 and parts[3] == "events":
                        events = []
                        for record in records:
                            from .evidence import viewer_events
                            events.extend(viewer_events(record))
                        body, kind = json.dumps(events).encode(), "application/json"
                    else:
                        self.send_error(404)
                        return
                else:
                    self.send_error(404)
                    return
            except OSError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description="Read-only local episode viewer")
    parser.add_argument("--root", type=Path, default=Path("trajectories/episodes"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    server = serve(args.root, args.port)
    print(f"Viewer: http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
