# TinyCC and arena restrictions

## Compiler boundary

The Judge builds generated C with TinyCC inside a disposable Docker container.
The pinned local sandbox image contains TinyCC and C development headers.
The candidate never invokes TinyCC itself. Build diagnostics are retained as
Judge evidence. Code should target ordinary C/POSIX interfaces supported by
that image; test compiler-specific behavior against the actual image before
relying on it. The legacy host TCC judge remains for old prototype commands
and must not be used for new generated-C benchmark evaluations.

## Runtime boundary

The base filesystem is read-only. Candidate code has no host file mounts,
general network, package manager, interpreter, compiler, or external helper
programs. Only the standard C/POSIX runtime supplied by the sandbox is
permitted; no external C libraries may be added. Output, memory, CPU, PID,
temporary storage, and time are bounded. The HTTP Judge communicates only
through a local Unix-domain socket. Large generated inputs belong in a seed
and generator recipe in evidence, not repeated large artifact files.
