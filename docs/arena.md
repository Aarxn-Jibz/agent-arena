# Arena contract (version 1)

The existing `arena run`, `solve`, and `marl` commands are a small C-task
prototype. The generic benchmark contract and Docker runner are infrastructure
for later experiments; they are not yet wired into those commands. In
particular, the old `arena run` path executes on the host. Use
`arena.sandbox.run_c` for generated C that needs isolation.

## Roles and authority

- **Challenger:** proposes a benchmark challenge and may explain why it probes
  the Solver. It can vary any parameter the benchmark validator permits. The
  arena does not prescribe its generation strategy.
- **Solver:** receives the validated requirements and may replace or rewrite
  its C implementation freely. It returns its response and candidate or patch.
- **Judge:** immutable for an experiment run. It validates challenges, builds
  and runs candidates, computes correctness and performance, applies the
  benchmark's acceptance and reward rules, and emits evidence. Neither agent
  may edit the Judge or decide its own score.

Benchmark documents specify observable requirements, allowed inputs, limits,
metrics, and acceptance rules. They do not suggest an implementation method.

## Episode lifecycle

1. Record run ID, global episode ID, seeds, benchmark version, container image
   digest, and Git commit before the episode.
2. Challenger submits a request and optional rationale. The benchmark validates
   it before any Solver work. An invalid challenge is rejected with a reason;
   it does not become a scored Solver failure.
3. Generate test inputs deterministically from the validated challenge,
   generator configuration, and seed. Store those settings, rather than large
   reproducible inputs, in episode evidence.
4. Solver submits a candidate. The Judge builds and tests it in the isolated
   runner. A build failure, correctness failure, or missed acceptance threshold
   rejects the candidate while retaining its feedback and measurements.
5. On acceptance, a coordinator may commit the candidate and record the new
   commit. On rejection, the accepted implementation and Git lineage stay at
   the prior commit. The Judge's verdict, not an agent claim, controls this.
6. Retain JSON evidence, readable summary, and viewer events. Only after the
   verdict and evidence are final may either policy update. Record rewards for
   both agents even when a candidate is rejected. The update method is outside
   the Judge contract.

An episode record is immutable once published. A retry is a new candidate
event or episode, with its own evidence; it must not overwrite an earlier
verdict. Keep accepted commits reachable. Store exact patches and source text
plus hashes so a reviewer can reconstruct changes. Do not let either agent
write directly to the accepted branch.

## Benchmark categories

| Purpose | Benchmarks | Role in evaluation |
| --- | --- | --- |
| Main training | compression engine; CSV/data CLI; HTTP server; expression evaluator; graph-processing CLI | Repeated challenges and policy updates are allowed. |
| Secondary validation | cache engine; log analyzer; search/indexing engine | Measure transfer with a fixed protocol; do not update policies on these results. |
| Held-out bosses | mini shell; tiny filesystem | Reserve for final evaluation, separate from training and tuning. |

No full benchmark in this table is implemented yet. The interface in
`arena/benchmark.py` is independent of task domain: initialize a benchmark,
validate a challenge, generate cases, judge correctness and performance,
provide reward inputs, and summarize. `arena.sandbox.run_c` owns the TCC build
and returns compiler output, process output, status, and elapsed time. The
benchmark owns its metric definitions and acceptance thresholds. It should
use fixed seeds and document its generator version.

## C/POSIX and offline operation

Candidates may use standard C and POSIX interfaces available in the pinned
sandbox image, including file and process APIs within limits. The only writable
path at runtime is `/scratch`. The Judge's `/work` holds input, binary, and
result files and is inaccessible for candidate writes. The container has no
network; external network calls are forbidden. Candidates may not invoke image
tools, package managers, compilers, interpreters, or external libraries beyond
the standard system C/POSIX runtime. The image makes tool directories
inaccessible to the candidate user. The LLM and policy processes remain on the
host, outside the execution container.

Build the image once with `docker build -f Dockerfile.sandbox -t
honours-arena-sandbox:local .`. The build needs package access; subsequent
evaluation uses `--pull=never` and requires no network. A pinned image digest
should be captured in every real run's configuration. The Python caller needs
permission to access the Docker daemon; Docker access itself is privileged
and should be limited to the trusted Judge process.

Default limits: one CPU quota, 256 MB RAM with swap disabled, 64 PIDs, two
temporary filesystems totaling 128 MB, 3-second candidate timeout, 10-second
build timeout, and 64 KiB returned per output stream. All are configurable by
`SandboxConfig`. The container is read-only apart from those tmpfs mounts,
uses no host bind mounts, grants the trusted Judge only the UID/GID, filesystem
access, and signal capabilities needed to manage an unprivileged candidate,
runs candidate code as UID 65534 without those capabilities, and is removed
after the evaluation. The scratch tmpfs cannot execute files. Tests cover
network, filesystem, host-file/tool, memory, PID, timeout, and output limits.

Docker limits and adversarial tests reduce risk but are not a proof against
kernel or container-runtime vulnerabilities. The legacy host judge remains
unsandboxed until explicitly migrated.
