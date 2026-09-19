# Lossless compression benchmark

The candidate is a C program built with TinyCC. Invoke it as
`solution compress` or `solution decompress`. It reads arbitrary bytes from
standard input and writes result bytes to standard output. Standard error may
carry diagnostics; successful operations exit with status 0. No extra text may
appear on standard output.

The candidate chooses its own compressed representation. For every valid
input, `decompress(compress(input))` must reproduce the original byte sequence
exactly, including zero bytes and empty input. A nonzero exit, crash, timeout,
or changed byte fails that case. The compressed output may be larger than the
input; correctness is judged before size or speed.

Validated challenges select one or more public input distributions, a byte
size from 0 to 8192, a 32-bit seed, and 1 to 3 repetitions. Public
distributions are repetitive, deterministic text, structured source text,
structured records, mixed patterns, high-entropy-like bytes, binary-oriented
bytes, and edge inputs. Inputs are regenerated from the challenge recipe and
seed; their content is not part of the Solver prompt.

The Judge measures original and compressed bytes, ratio, bytes saved or
expanded, median compression and decompression time, and resource status.
Memory is reported only when reliably available. A challenge passes only when
every generated case round-trips byte-for-byte. Performance reward is eligible
only after full correctness. No external libraries, utilities, network, or
host files are available to the candidate.
