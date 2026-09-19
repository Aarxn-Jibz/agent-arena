# Local evidence viewer

Run `python3 -m arena.viewer --root trajectories/episodes --port 8765`, then
open `http://127.0.0.1:8765`. The server binds only to localhost, needs no
network or frontend build tools, and never writes experiment files. Point
`--root` at the directory passed to `write_episode`.

The viewer reads complete version-1 episode JSON files and ignores a JSON file
that is still being written. It refreshes the run and history list every three
seconds. Challenger, Solver, and Judge details use the generic evidence keys;
benchmark-specific metrics are shown as structured data. Source diffs are
computed against the previous accepted episode in the same run. A first
episode has no previous accepted source. The `/api/runs/<id>/events` endpoint
exposes the same ordered viewer event contract for future clients.
