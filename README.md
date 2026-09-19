# Agent Arena

The future generic benchmark and sandbox contracts are in
[docs/arena.md](docs/arena.md) and [docs/evidence.md](docs/evidence.md).
The existing CLI paths below remain the original small C-task prototype;
they are not yet connected to the new Docker runner.
The current benchmark suite, evaluator method, and remaining gaps are in
[docs/research.md](docs/research.md). The read-only local viewer is documented
in [docs/viewer.md](docs/viewer.md).

A minimal autonomous C-programming environment where a small language model
generates C programs, receives deterministic compiler/test feedback and
rewards, retries solutions, and records trajectories for future
reinforcement-learning experiments. Submitted C is compiled with **TCC** only,
run against deterministic hidden tests, scored with a simple reward, and logged
as JSONL trajectories.

This is a submission-ready prototype for the honours project. It is a
**trajectory-collection environment**: the model runs in inference mode and no
weight updates or reinforcement-learning training have been performed yet.

## Research motivation

- Small language models (such as the 360M-parameter SmolLM2) are often weak and
  inconsistent programmers, and they struggle to follow strict exact-output
  constraints.
- C is a useful test bed because correctness can be evaluated **objectively**:
  a program either compiles (or not) and either passes deterministic tests (or
  not).
- The environment provides machine-verifiable feedback and reward signals at
  every step (compiler diagnostics, test results, rewards), which an agent can
  use to improve its next attempt.
- The collected trajectories are intended to support **future** policy
  optimization and adversarial MARL between a Challenger (problem writer) and a
  Solver (programmer).

## Current architecture

```
Problem
   |
   v
SmolLM2 Solver
   |
   v
Generated C
   |
   v
TCC compiler
   |
   +--> compiler feedback
   |
   v
Deterministic tests
   |
   +--> stdout/stderr/runtime
   |
   v
Reward
   |
   +--> trajectory JSONL
   |
   +--> feedback to Solver for retry
```

## Implemented features

- **TCC-only compilation** (`arena/runner.py`): the single C compiler; captures
  compiler exit code, stdout, and stderr.
- **Execution timeout**: per-task `timeout_seconds`; infinite loops are killed
  and reported.
- **stdin/stdout test cases**: each task carries test cases with input and
  expected output; output is compared after a simple normalization (CRLF -> LF,
  trailing whitespace stripped per line, trailing blank lines dropped).
- **Runtime measurement**: wall-clock timing via `time.perf_counter()`.
- **Deterministic reward** (`arena/evaluate.py`): one reward value per attempt
  (see [Reward](#reward)).
- **JSONL trajectory logging** (`arena/log.py`): one JSON object per attempt,
  including generated source, full evaluation result, reward, and metadata.
- **CLI** (`arena/cli.py`): `python -m arena run` (deterministic judge) and
  `python -m arena solve` (LLM retry loop).
- **Solver retry loop** (`arena/solver.py`): model-agnostic; returns
  compiler/test feedback to the model and retries up to a configurable cap.
- **Tabular MARL prototype** (`arena/marl.py`): independent Q tables select one
  of three existing tasks and one of three prompt strategies. The SmolLM2
  model stays frozen. Episode JSONL, Q state, and a Markdown run report are saved.
- **Hugging Face SmolLM2-360M-Instruct integration** (`arena/solver_llm.py`,
  optional dependency): CPU-inference adapter, chat-template based, hidden
  tests never shown to the model.
- **Automated tests**: `unittest` suite (61 tests) exercising the arena with
  the real TCC binary.

## Not implemented / future work

Explicitly **not** implemented — these are future work, not current behaviour:

- Challenger task generation (the current policy only selects existing tasks);
- LoRA, PPO, GRPO, or any reinforcement-learning weight updates;
- SmolLM2 policy optimization or Solver/Challenger model co-training;
- learned curriculum.

No model weights have been updated; the model runs in inference mode only.

## Setup (WSL/Linux)

Requirements: Linux/WSL, Python 3.10+, and the `tcc` binary.

### 1. Python virtual environment

The deterministic arena needs only the standard library. The optional LLM
Solver additionally needs `torch` + `transformers` (a CPU build is enough):

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers
```

### 2. TCC

```bash
sudo apt-get install tcc
```

Verify TCC:

```bash
tcc -v          # e.g. "tcc version 0.9.27 (x86_64 Linux)"
```

> The arena resolves `tcc` on `PATH` (override with `TCC_BIN=/path/to/tcc`).
> This repository was developed on a machine without passwordless sudo, so TCC
> 0.9.27 was bootstrapped into `~/.local/bin` from the official source
> (`https://download.savannah.gnu.org/releases/tinycc/tcc-0.9.27.tar.bz2`),
> patching `lib/bcheck.c` to disable the malloc hooks on glibc >= 2.34. If your
> `tcc` is not on `PATH`, export `PATH="$HOME/.local/bin:$PATH"` or set
> `TCC_BIN`. **TCC is the only C compiler used by the arena.**

### 3. Run the tests

```bash
python3 -m unittest discover -s tests     # 61 tests, uses the real tcc binary
```

## Demo commands

### Deterministic arena demo (no LLM, works offline)

```bash
python3 -m arena run tasks/001_add.json solutions/add.c       # 4/4 pass, reward 16.0
python3 -m arena run tasks/001_add.json solutions/bad.c       # compile error + diagnostics
python3 -m arena run tasks/001_add.json solutions/wrong.c     # compiles, 1/4 pass
python3 -m arena run tasks/001_add.json solutions/infinite.c  # timed out, penalty applied
python3 -m arena run --json tasks/001_add.json solutions/add.c   # full result as JSON
python3 -m arena run tasks/001_add.json solutions/add.c \
  --log trajectories/demo.jsonl --attempt 1                     # append trajectory record
```

Exit code is `0` on pass, `1` on fail, `2` on usage/io error.

### Real-model Solver demo (optional, needs `.venv` + Hugging Face access)

```bash
.venv/bin/python -m arena solve tasks/001_add.json --attempts 3 --log trajectories/demo.jsonl
```

This runs the full verified path: task -> SmolLM2 -> generated C -> TCC ->
tests -> feedback -> retry -> reward -> trajectory JSONL. On first run the
model downloads roughly 700 MB (about 7 minutes); later loads are cached
(about 30-60 s). If model loading fails tomorrow, the deterministic demo above
still demonstrates everything except the LLM step.

### MARL policy selection (optional; invokes SmolLM2)

```bash
.venv/bin/python -m arena marl --episodes 9 --attempts 2
```

The command prints one progress line per episode and a run summary. It writes
`marl_state.json`, `trajectories/marl.jsonl`, and
`trajectories/marl-report.md` by default. Reusing the same state path resumes
with global episode numbers and the saved policy RNG state; use the same seed
and hyperparameters. `--state`, `--log`, and `--report` set artifact paths.
The Markdown report covers the episodes in the current invocation, while the
state and JSONL log continue across invocations. This selects tasks and prompts;
it does not train SmolLM2 weights or establish general improvement.

## Task format

```json
{
  "id": "001",
  "title": "Add two integers",
  "prompt": "Read two integers a and b from stdin and print their sum.",
  "timeout_seconds": 2,
  "tests": [
    {"input": "2 3\n", "expected_output": "5\n"},
    {"input": "-5 12\n", "expected_output": "7\n"}
  ]
}
```

Required fields: `id`, `title`, `prompt`, `timeout_seconds` (positive number),
and a non-empty `tests` list where every test has `input` and
`expected_output`. Three sample tasks ship in `tasks/` (add, max of array,
reverse string).

## Reward

For one evaluated attempt:

```
reward = 1 (compiled)
         + 10 * passed/total
         + 5  (all tests pass, no timeout)
         - 5  (any timeout)
         - 1  per test that exits with a non-zero code (and did not time out)
         - 0.5 * (attempt - 1)      <-- retry penalty beyond the first attempt
```

Compile failures score `0.0`. A test passes only if the program exits 0, does
not time out, and its normalized stdout equals the normalized expected output.
The mapping from (source, task, attempt) to reward is deterministic — the same
input always yields the same reward.

## Real results (SmolLM2-360M-Instruct, CPU)

Configuration: temperature 0.9, top_p 0.9, at most 512 generated tokens,
maximum 3 attempts. The prompt hides expected test outputs and shows one
clearly labeled *different*-task exemplar before the real task.

| Task | Result | Attempt progression | Rewards |
| --- | --- | --- | --- |
| 001 Add two integers | SUCCESS | pass on attempt 1 (4/4 tests) | [16.0] |
| 002 Max element of array | FAILURE | 1/5 -> 4/5 -> compile error (missing `<limits.h>`) | [3.0, 8.5, 0.0] |
| 003 Reverse a string | FAILURE | 0/4, 0/4, 0/4 (repeated non-reversing solution) | [1.0, 0.5, 0.0] |

Trajectories: `trajectories/solver_001.jsonl`, `solver_002.jsonl`,
`solver_003.jsonl`.

Interpretation (no causal claims beyond the data):

- Task 001 proves full successful end-to-end execution: model -> generated C ->
  TCC -> tests -> reward -> trajectory, with reward 16.0.
- Task 002 demonstrates feedback-associated improvement from 1/5 to 4/5 tests
  before a regression (a compile error) on the final attempt.
- Task 003 demonstrates a limitation: the model fails to effectively use
  feedback, repeating the same non-reversing solution.
- Together these outcomes motivate future policy optimization on the collected
  trajectories.

## Performance observation

- Model generation: about 10 seconds per reply (~90-130 tokens) on CPU.
- TCC compile + 4-test evaluation: well under ~10 ms for the measured cases.
- First model load/download: ~7 minutes including roughly 700 MB download;
  cached subsequent load: ~30-60 s.

LLM inference therefore dominates the runtime of the current prototype; the
deterministic judging step is negligible in comparison.

## Limitations

- Tiny 360M-parameter model.
- CPU-only, fp32 inference (no quantization).
- Stochastic generation (light sampling), so results vary between runs.
- Small, fixed sample-task set (3 tasks).
- No model weight updates or fine-tuning; only task/prompt Q tables update.
- TCC is the only supported C compiler.
- Current tests/rewards are task-specific deterministic evaluation, not a
  general-purpose verifier.
- No task-generating Challenger agent yet; the policy selects fixed tasks.

## Future task-generating MARL design (conceptual, not implemented)

```
  Challenger --> task --> Solver
      ^                    |
      |                    v
      +-- reward <--- Judge ---> Solver reward
```

- The Solver would optimize task-solving performance against the deterministic
  judge.
- The Challenger would seek valid tasks near the Solver's capability frontier
  (hard but solvable).
- The deterministic judge remains the authoritative correctness signal in any
  such design.
- The current tabular policies update task and prompt choices; task-generating
  agents and model-weight updates remain future work.
