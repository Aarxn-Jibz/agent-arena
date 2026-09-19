# Episode evidence and viewer event contract (version 1)

`arena.evidence.write_episode(root, record)` writes
`<root>/<run_id>/<episode_id:06d>.json`, a matching Markdown summary, and
appends three chronological records to `events.jsonl`. Duplicate episode IDs
are rejected. Run IDs are path-safe. This contract is for the future generic
arena; current `marl.jsonl` retains its older format.

Required episode fields:

| Field | Meaning |
| --- | --- |
| `run_id`, `episode_id`, `benchmark` | Stable identity and global chronology. |
| `seeds`, `input_generation` | Policy/input seeds plus generator name, version and configuration. Store large reproducible inputs by recipe. |
| `git_before`, `git_after` | Accepted commit before the episode and new commit only on acceptance. |
| `challenger` | `request` and optional `rationale`. |
| `solver` | `response`, complete `candidate`, optional `patch` against previous accepted source. |
| `build` | TCC `exit_code`, `stdout`, and `stderr`. |
| `correctness`, `performance` | Benchmark-defined structured measurements; include case-level evidence as needed. |
| `judge` | Objective `feedback` and optional `resource_usage`. |
| `rewards` | Numeric `solver` and `challenger` values assigned by the Judge. |
| `outcome` | `accepted` or `rejected`. |
| `timestamps` | ISO-8601 `started_at`, `candidate_at`, `finished_at`. |
| `hashes` | Added by the writer: SHA-256 of challenge, candidate, and patch when present. |
| `reference_reads` | Optional list of `{document, section, read_at}` for local reference material consulted by an agent. |

The writer adds `schema_version: 1`. A real run should also retain an immutable
run configuration containing benchmark and generator versions, model revision,
Judge commit, container image digest, and reward/acceptance settings. Artifact
paths and their hashes can be added to the episode record. The candidate and
patch are retained even for rejection; compiler and Judge output are retained
without relying on an agent's summary. A rejected record has `git_after: null`.

`events.jsonl` is the viewer feed. Every event has `schema_version`, `run_id`,
`episode_id`, `actor`, `kind`, and `at`. The fixed order is:

1. `challenger/challenge`: request and rationale.
2. `solver/candidate`: response, complete code, patch, and reference reads.
3. `judge/verdict`: correctness, performance, resource usage, feedback,
   rewards, and accepted/rejected outcome.

Consumers sort by global episode ID, then this actor order; timestamps show
wall-clock chronology. The viewer must treat code, prompts, compiler output,
and feedback as untrusted text and escape them on display. It may render a diff
from the saved patch or compare the candidate with the previous accepted Git
commit. The JSON episode remains authoritative if an event stream is damaged.
