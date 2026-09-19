# Permitted POSIX interfaces

## Files and processes

Within the sandbox, `open`, `read`, `write`, `close`, `lseek`, and `fstat` are
available. `read` and `write` may transfer fewer bytes than requested; loop as
needed and handle interruption. `fork`, `exec`, `waitpid`, and signals exist
within the PID and time limits, but candidate code cannot launch external
image tools. The filesystem is read-only except for the disposable scratch
area. No host experiment path is mounted.

## Sockets

`socket`, `bind`, `listen`, `accept`, `connect`, `send`, and `recv` are available
for the HTTP benchmark's local Unix-domain socket. The candidate receives its
socket path as an argument. Internet access is disabled. A successful `send`
or `recv` may transfer fewer bytes than requested. A peer may close early.

## Timing and resource APIs

`clock_gettime` can read monotonic time for internal measurements. The Judge's
measurement, not a candidate-reported clock, determines score. `getrusage`
can report process usage where supported, but the arena currently reports
memory only when its Judge has a reliable measurement. Container CPU, memory,
PID, output, and wall-time limits still apply regardless of candidate timing.
