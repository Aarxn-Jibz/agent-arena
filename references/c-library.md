# C standard library and binary I/O

## Streams

`fread(ptr, 1, n, stream)` returns the number of bytes actually read; zero
may mean EOF or an error. `ferror` and `feof` distinguish them. `fwrite` also
returns a count and may write fewer bytes than requested. Standard input and
output can carry zero bytes; do not route binary data through string functions.
`fflush(stdout)` makes buffered output available before waiting for input.

## Formatting

`snprintf` reports how many characters would have been written, excluding the
terminating zero. Check for truncation. `strtol` and related functions offer
end-pointer and range checks that a blind numeric conversion lacks. Exact
arena output should not include prompts or diagnostics on standard output;
use standard error for diagnostics.

## Memory and error handling

`malloc`, `calloc`, `realloc`, and `free` manage storage. Check allocation
results. `errno` is meaningful only when an API documents an error and reports
failure; save it before another call may change it. Close opened streams and
descriptors. Handle partial input, empty input, and early EOF explicitly.
