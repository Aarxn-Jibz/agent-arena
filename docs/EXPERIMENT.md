# Offline autonomous experiment

Run from the repository root using the existing Python environment. The five main arenas are `compression,csv,http,expression,graph`. `cache,log,search` are available by explicit selection for separate validation runs. The held-out mini shell and tiny filesystem are **excluded** from this command and remain sealed for the first frozen zero-shot comparison.

The coordinator loads one frozen `HuggingFaceTB/SmolLM2-360M-Instruct` model and uses separate Challenger and Solver prompts. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set before loading; `local_files_only=True` prevents fetching. It records the exact cached snapshot revision in the manifest and loads that same revision on resume. The model is never trained. Adaptation here is in the last-five scratchpads, accepted source project, and an optional benchmark-selection Q policy. This reuses the existing Q helpers with the generic benchmark names, not the old three-task action list.

## Setup and preflight

The official model and tokenizer must be cached once. In a connected setup environment, run `.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("HuggingFaceTB/SmolLM2-360M-Instruct")'`. This is the only permitted model acquisition step; no model download occurs in the experiment. Verify with:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m arena experiment --preflight
```

Preflight loads the cached model and runs a tiny C program only inside the local Docker/TinyCC sandbox image. The real run also uses Docker `--pull=never --network=none`, a read-only base image, no host mounts, CPU/memory/PID/tmpfs/timeout limits, and bounded output. The localhost viewer is separate and read-only. Python agents receive no network tool.

## Episode and state

Each episode starts from the accepted Git worktree on `experiment/<run-id>`; the main branch stays untouched. Challenger sees a public benchmark specification, a validated schema example, recent Judge outcomes, and its own last five short interpretations. It has two chances to produce a valid JSON challenge; invalid outputs are evidence, and a validated deterministic fallback follows. Solver sees the public specification, validated challenge, selected source files, latest accepted diff, recent failures, selected local reference excerpt, and its own last five interpretations. It returns JSON with a short summary and changed files. Paths are restricted to `solution.c` and `src/<safe-name>.c/.h`; the complete project is capped at 3 MiB. Large source trees are listed and selected lexically. Source and reference reads are recorded. Output is constrained by token limits and the tokenizer's measured 8192-token context. One continuation is allowed only for a truncated JSON response.

The immutable benchmark Judge validates and executes candidate C **only** in the Docker/TinyCC sandbox. Candidate and accepted baseline are measured on the same challenge. First full-correctness candidates may be accepted. Subsequent candidates must retain full correctness and exceed a deterministic performance threshold: generally at least 10% and 5 ms faster. Compression may alternatively improve compressed size by at least 5% and 16 bytes without a material speed regression; speed improvements cannot cost more than 5% and 16 compressed bytes. Two recent accepted challenges for that benchmark are replayed to catch regressions. These are conservative local decisions, not claims of statistical performance or model improvement. Rejected candidate source, patch, response, scores, and memories remain in episode evidence; only accepted candidate projects receive Git commits.

`trajectories/episodes/<run-id>/` holds `manifest.json`, atomic `state.json`, one JSON and Markdown file per completed episode, `events.jsonl`, `report.md`, and a dedicated Git worktree. `pending.json` marks a transaction before Solver/Judge work. An interrupted unjudged attempt is archived as `interrupted-*.json` and is never accepted. If a commit happened just before interruption, resume checks its parent and message, completes evidence, and checkpoints the accepted HEAD. The event stream is rebuilt from complete episode JSON on resume. Episode seeds derive from the saved run seed and global episode number. Each model generation seeds an isolated PyTorch RNG scope. Resume checks model revision, benchmark list, and resource/token/quota settings. It retains at most five compact memory items per agent; these interpretations are not Judge facts.

`--selection-mode adaptive` is the default for a new run. It applies epsilon-greedy (`epsilon=0.3`) selection over the configured benchmark names. The three states are the existing low/medium/high buckets of recent correctness. The chosen benchmark's Q value receives the existing tabular update (`alpha=0.4`, `gamma=0.8`) from a frontier reward computed from Judge pass fraction; invalid Solver output gets zero policy reward. This is a learned **Challenger benchmark-selection policy**, not SmolLM2 weight training or a full two-learner MARL algorithm. Its Q table, recent performance, and RNG state are checkpointed and included in episode evidence. Use `--selection-mode round_robin` for deterministic coverage without policy selection. The pilot used round robin.

The default soft artifact limit is 15 GiB and hard limit is 20 GiB, excluding shared model weights and Docker image storage. It counts run-directory files and the uncompressed sizes of Git objects introduced on the experiment branch. Temporary files are removed after the soft limit; deterministic corpora are generated in memory from recipe and seed and never retained. No new episode starts within a 64 MiB safety margin of the hard limit. Candidate projects are capped at 3 MiB, compiler/program output at 1 MiB each, and episode evidence is capped at 20 MiB. Each safe episode is allowed to finish after the wall-clock deadline; the next one is not started. After every episode, evidence, accepted Git HEAD, and memories are checkpointed. The report states policy/source outcomes, not model improvement.

## Commands

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m arena experiment --run-id overnight-20260919 --hours 10 --seed 42 --benchmarks compression,csv,http,expression,graph --selection-mode adaptive --soft-gb 15 --hard-gb 20
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m arena experiment --run-id overnight-20260919 --hours 10 --seed 42 --benchmarks compression,csv,http,expression,graph --selection-mode adaptive --soft-gb 15 --hard-gb 20 --resume
.venv/bin/python -m arena.viewer --root trajectories/episodes --port 8765
```

Viewer URL: `http://127.0.0.1:8765/`. Do not use the legacy `arena run/solve/marl` commands for generated C in this experiment; they use the older host judge. Start the overnight command manually only after reviewing a pilot.
