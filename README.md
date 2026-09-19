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
- Small CLI and 3 sample tasks.

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
| `arena/cli.py`, `arena/__main__.py` | `python -m arena` command line |
| `tasks/*.json` | sample tasks (add, max of array, reverse string) |
| `solutions/*.c` | demo solutions incl. compile-error / wrong / infinite cases |
| `tests/` | `unittest` suite (52 tests) |

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

No Python dependencies outside the standard library are required.

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

## Deliberate future work (not implemented)

- **Solver agent**: an LLM (targeting `HuggingFaceTB/SmolLM2-360M-Instruct`)
  generates C from the task prompt, receives compiler/test feedback, and
  retries until success or a max-attempt cap — the deterministic arena above
  is the judge.
- **Challenger agent**: LLM-generated problems with deterministically validated
  test cases.
- **Adversarial MARL / PPO / GRPO / fine-tuning / policy optimization**: none of
  this exists yet. The arena only produces data.

Everything currently in the repository is deterministic: an evaluation of the
same (source, task) yields the same reward, the same pass/fail results, and the
same trajectories except for wall-clock timing fields.