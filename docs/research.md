# Honours experiment platform: evaluator guide

## Research question and present learning boundary

The project asks whether interacting Challenger and Solver policies can use
objective programming feedback to choose more useful challenges and responses
over repeated episodes. The current `arena marl` prototype updates small task
and prompt-selection Q tables around a **frozen** SmolLM2 model. It does not
train SmolLM2 weights, establish general improvement, or yet drive the new
benchmark suite. A separate future study could adapt model weights and compare
that intervention against the frozen-model policy baseline, with its own
checkpoints and held-out evaluation.

## Roles and sequence

The Challenger identifies weaknesses and proposes valid pressure within a
benchmark schema. The Solver sees the public specification, validated
challenge, current accepted project state, permitted local references, and
Judge feedback. It may rewrite its solution freely. The immutable Judge
validates the challenge, builds and evaluates the candidate, measures
correctness and performance, determines rewards and acceptance, and retains
evidence. Agents cannot decide objective results or alter the Judge.

One episode is: record accepted Git commit and seeds; validate challenge;
generate deterministic inputs; get candidate; evaluate in Docker; record
verdict and evidence; commit an accepted candidate through the trusted
coordinator; then update agent policies. Rejected candidates leave the
accepted implementation unchanged. Both accepted and rejected attempts are
retained because failures expose capability boundaries and provide feedback
for later policy analysis. The current benchmark APIs implement evaluation and
evidence handoff, not an autonomous multi-agent coordinator.

## Benchmark taxonomy

| Group | Benchmarks | Policy use |
| --- | --- | --- |
| Main training | [Compression](benchmarks/compression/SPEC.md), [CSV/data](benchmarks/csv/SPEC.md), [HTTP server](benchmarks/http/SPEC.md), [expression evaluator](benchmarks/expression/SPEC.md), [graph CLI](benchmarks/graph/SPEC.md) | May generate training episodes and policy updates. |
| Secondary validation | [Cache](benchmarks/cache/SPEC.md), [log analyzer](benchmarks/log/SPEC.md), [search/indexing](benchmarks/search/SPEC.md) | Evaluate transfer with a fixed protocol; do not train on these results. |
| Held-out bosses | [Mini shell](benchmarks/mini_shell/SPEC.md), [tiny filesystem](benchmarks/tiny_filesystem/SPEC.md) | First use is zero-shot; baseline and trained evaluations are frozen before any adaptation. |

Public specifications define observable behavior and challenge bounds, never
an algorithm, data structure, or implementation progression. Hidden inputs are
regenerated from versioned recipes and seeds by the Judge; they must not be
included in training prompts or evidence.

## Sandbox and offline operation

New benchmark-generated C runs **only** through `arena.sandbox.run_c` in a
fresh Docker container. The existing `arena run`, `solve`, and `marl` commands
still use the old host judge for their original sample tasks and must not be
used to evaluate new generated benchmark candidates. The container has no
general network or host experiment mounts, a read-only base filesystem,
bounded temporary storage, CPU, memory, PIDs, output, and time. The HTTP Judge
uses only an in-container Unix-domain socket. The LLM and policy processes
stay outside Docker. The trusted Judge image contains TinyCC, but candidate
programs cannot invoke image tools. Candidates may use only the permitted
standard C/POSIX runtime; no external C library or Internet service is
available. The image must be built before an offline run, and its digest
recorded in the run configuration.

Docker isolation reduces risk, but Docker daemon access is privileged and
must stay with the trusted Judge. The current limits do not constitute a
formal sandbox proof. Memory is reported as unavailable where the Judge does
not have a reliable per-candidate measurement.

## Evidence, Git, viewer, and reproducibility

The [episode schema](evidence.md) retains the run/global episode ID, benchmark,
seeds, generator recipe, Challenger request/rationale, Solver response/source
and patch, compiler output, correctness/performance, Judge feedback, rewards,
verdict, timestamps, relevant hashes, and before/after Git commits. Large
corpora are regenerated rather than copied into each episode. The accepted
commit identifies the program state before an episode; a rejected episode has
no new accepted commit. The [local viewer](viewer.md) reads complete evidence
files and events without modifying experiments, showing the Challenger →
Solver → Judge story and chronological history. It tolerates a partially
written JSON episode during a live run.

For reproducibility, freeze benchmark and generator versions, Judge commit,
container image digest, model revision, prompts, seed, resource limits, and
reward rules in the run configuration. `arena marl` persists its Q tables,
global episode number, recent performance, and policy RNG state for
episode-boundary continuation. Wall-clock performance has normal variation;
use repeated, paired comparisons before interpreting small differences.

The [local reference pack](../references/README.md) is frozen per run. Future
agents can request individual documents and sections; evidence can record
each reference read and timestamp. Reference material explains APIs, not
benchmark solutions.

## What is and is not demonstrated

The deterministic Judges, Docker sandbox, benchmark generators/oracles,
evidence writer, and read-only viewer have automated tests. Small integration
candidates exercise selected paths. There has been no long SmolLM2 benchmark
run in this milestone, no trained model weights, no autonomous accepted-commit
coordinator, and no held-out policy comparison. An accepted integration test
candidate proves a contract path, not research improvement.
