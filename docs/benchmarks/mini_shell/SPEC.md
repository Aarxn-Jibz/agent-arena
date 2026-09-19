# Held-out boss: mini shell

This boss is reserved for a first **zero-shot** comparison of the original
baseline system and the trained system. Neither may adapt on boss results
until both first-use results are frozen outside training evidence.

The candidate is a TinyCC-built C program. It reads a script of one command
per LF-terminated line and writes exact command output to standard output.
Commands run in order. Public command vocabulary:

| Command | Observable behavior |
| --- | --- |
| `echo TEXT` | Write `TEXT` and LF. |
| `pwd` | Write current virtual directory and LF; initially `/`. |
| `cd /` or `cd /work` | Change virtual directory, no output. |
| `set name VALUE` | Store the value, no output. |
| `get name` | Write stored value or an empty line. |
| `run true` / `run false` | Complete a child-like command with status 0 / 1, no output. |
| `run echo TEXT` | Write `TEXT` and LF; child-like status 0. |
| `status` | Write the most recent child-like status and LF; initially 0. |
| `exit N` | Stop with process exit code N. |

The specification defines observable command and process-status semantics,
not a required internal mechanism. No external programs, Internet, or host
files are available. Validated hidden challenges select 1–50 commands, seed,
and time limit. The Judge checks exact output, exit code, timeout, and runtime.
Hidden workloads are retained only in the sealed boss evaluation area.
