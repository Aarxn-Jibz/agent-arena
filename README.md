# Agent Arena

A minimal adversarial C-programming learning environment: submitted C source is
compiled with **TCC**, executed against deterministic tests, scored with a
simple reward, and logged as trajectories for future policy optimization.

This is a prototype for the honours project. The long-term goal is to collect
interaction trajectories for adversarial MARL / policy optimization between a
*Challenger* (writes problems) and a *Solver* (writes C code). **No training of
any kind has happened yet** — this repository is the deterministic environment
that will produce the data.

## What is implemented

- Compile C source with TCC (`tcc solution.c -o solution`), capturing
  exit code, stdout and stderr of the compiler.
- Run the binary per test case with a configurable timeout
  (`time.perf_counter()` timing, infinite loops are killed).
- Deterministic test evaluation: stdin -> program -> stdout, compared via a
  simple normalization (CRLF -> LF, trailing whitespace stripped per line,
  trailing blank lines dropped). A test passes only if the program exits 0,
  does not time out, and its normalized stdout matches exactly.
- Deterministic reward:
  ```
  reward = 1 (compiled) + 10 * passed/total
           + 5 (all tests pass, no timeout)
           - 5 (any timeout)
           - 1 per test that exits non-zero
           - 0.5 per attempt beyond the first
  ```
  Compile failures score 0.
- JSONL trajectory logging (one JSON object per attempt).
- Small CLI, 3 sample tasks, and demo solutions.
- **Solver agent** (experimental): `HuggingFaceTB/SmolLM2-360M-Instruct` generates
  C from the task prompt, the arena judges it with TCC, compiler/test feedback
  is returned to the model, and the loop retries up to a configurable cap.
  Generation uses light sampling (temperature 0.9, top_p 0.9, bounded at 512
  tokens) so retries can actually vary; greedy decoding repeatedly produced
  byte-identical, feedback-unaware answers on the small model.
- **Verified real runs** (see below): task 001 solved on attempt 1; task 002
  and 003 failed within 3 attempts — SmolLM2-360M does not yet reliably follow
  strict exact-output constraints, which is precisely the data this
  environment is meant to collect.

## Architecture

```
problem (task JSON)
   -> C source
   -> TCC compile (arena/runner.py)
   -> run against tests (arena/evaluate.py)
   -> reward (arena/evaluate.py compute_reward)
   -> retry / stop
   -> trajectory log (arena/log.py)
```

| File | Purpose |
| --- | --- |
| `arena/runner.py` | TCC compile + process execution with timeout |
| `arena/evaluate.py` | task loading, output normalization, test evaluation, reward |
| `arena/log.py` | JSONL trajectory logging |
| `arena/solver.py` | Solver retry loop (model-agnostic: feedback + retries) |
| `arena/solver_llm.py` | optional transformers adapter (SmolLM2-360M-Instruct) |
| `arena/cli.py`, `arena/__main__.py` | `python -m arena` command line |
| `tasks/*.json` | sample tasks (add, max of array, reverse string) |
| `solutions/*.c` | demo solutions incl. compile-error / wrong / infinite cases |
| `tests/` | `unittest` suite (61 tests) |

## Setup

Requirements: Linux/WSL, Python 3.10+, the `tcc` binary.

Install TCC (Ubuntu/Debian WSL):

```bash
sudo apt-get install tcc
```

> This repository was developed on a machine without passwordless sudo, so
> TCC 0.9.27 was bootstrapped into `~/.local/bin` from the official source
> (`https://download.savannah.gnu.org/releases/tinycc/tcc-0.9.27.tar.bz2`)
> with a one-line patch to `lib/bcheck.c` disabling the malloc hooks on
> glibc >= 2.34. If `tcc` is not on PATH, export
> `PATH="$HOME/.local/bin:$PATH"` or set
> `TCC_BIN=/path/to/tcc`. **TCC is the only C compiler used by the arena.**

No Python dependencies outside the standard library are required for the
deterministic arena. The optional LLM Solver additionally needs
`torch` + `transformers` (CPU build is enough); they are installed in the
project `.venv`:

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers
```

## Usage

```bash
# Run all tests
python3 -m unittest discover -s tests -v

# Evaluate a solution and print a summary
python3 -m arena run tasks/001_add.json solutions/add.c

# Full result as structured JSON
python3 -m arena run --json tasks/002_max_array.json solutions/max_array.c

# Append the attempt to a trajectory JSONL (retry loop friendly)
python3 -m arena run tasks/001_add.json solutions/add.c --log trajectories/train.jsonl --attempt 1

# Run the LLM Solver retry loop (hidden tests stay hidden from the model)
.venv/bin/python -m arena solve tasks/001_add.json --attempts 3 --log trajectories/solver_001.jsonl

# Exit code is 0 on pass, 1 on fail, 2 on usage/io error
```

## Demo

```bash
python3 -m arena run tasks/001_add.json solutions/bad.c       # compile error + diagnostics
python3 -m arena run tasks/001_add.json solutions/wrong.c     # compiles, 1/4 pass
python3 -m arena run tasks/001_add.json solutions/infinite.c  # timed out, penalty applied
python3 -m arena run tasks/001_add.json solutions/add.c       # 4/4 pass, reward 16.0
```

Attempt 1 (wrong solution) and attempt 2 (correct solution) against
`tasks/002_max_array.json` produce `reward: 1.0, success: false` then
`reward: 15.5, success: true`, exactly mirroring the intended Solver retry
loop — but driven by a submitted C file instead of a model.

## Task format

```json
{
  "id": "001",
  "title": "Add two integers",
  "prompt": "Read two integers a and b from stdin and print their sum.",
  "timeout_seconds": 2,
  "tests": [
    {"input": "2 3\n", "expected_output": "5\n"}
  ]
}
```

`python -m arena run` reports per-test stdout, stderr, exit code, timeout flag
and elapsed time in the JSON result (`tests[].*`) plus the last test's
`program_stdout` / `program_stderr` / `program_exit_code` at the top level.

## Experimental observations (SmolLM2-360M-Instruct, CPU)

All runs used greedy-bounded light sampling (temperature 0.9, top_p 0.9, at
most 512 generated tokens; sampling – not greedy – because greedy decoding
repeatedly returned byte-identical, feedback-unaware programs). The prompt is:
solver instructions, a clearly labeled **different**-task exemplar, the real
task exactly once, then a request for one C code block. Expected outputs of the
real task are never shown to the model; the TCC/test environment is the only
judge.

Verified real path: model → generated C → TCC compile → tests → compiler/test
feedback → retry up to 3 attempts → deterministic reward → JSONL trajectory.

- **task 001 (add two integers): a real successful rollout.** The model
  generated the minimal correct program on attempt 1; 4/4 tests passed, reward
  16.0, trajectory `trajectories/solver_001.jsonl`.
- **task 002 (max of array): failed within 3 attempts** — 1/5, then 4/5
  (subtle input-scanner bug), then a compile error (missing `<limits.h>`),
  rewards [3.0, 8.5, 0.0].
- **task 003 (reverse string): failed within 3 attempts** — the same
  non-reversing, labeled program repeated 3 times, rewards [1.0, 0.5, 0.0].
- Earlier repeated trials showed substantial stochasticity and inconsistent
  success for task 001 (success in some trials after 1-2 attempts, failure in
  others). The small model does not yet reliably follow exact-output
  constraints, which is precisely the kind of signal this environment is meant
  to collect.
- **No model weight training, fine-tuning, LoRA, or MARL/policy optimization
  has been performed.** The model runs in inference mode only; trajectories are
  collected for future work.

## Deliberate future work (not implemented)

- **Challenger agent**: LLM-generated problems with deterministically validated
  test cases.
- **Adversarial MARL / PPO / GRPO / fine-tuning / policy optimization**: none of
  this exists yet. The arena and Solver only produce data.

Everything in the deterministic arena is deterministic: an evaluation of the
same (source, task) yields the same reward, the same pass/fail results, and the
same trajectories except for wall-clock timing fields. LLM generation is
stochastic by design and is recorded verbatim in each attempt's trajectory.