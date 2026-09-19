# Log analyzer validation benchmark

The candidate is a TinyCC-built C program invoked as
`solution <query> <level> <source>`. Input is LF-terminated records
`timestamp|level|source|message`. Timestamps are eight decimal digits;
levels are `INFO`, `WARN`, or `ERROR`; sources are `svc0`–`svc2`.
Malformed lines are ignored.

Public queries: `count` writes the number of valid records at the requested
level and LF; `by_level` writes `INFO n`, `WARN n`, `ERROR n` on separate lines;
`filter` writes original valid lines matching both requested level and source,
in input order. No extra output is allowed.

Challenges vary record count 0–10,000, uniform/bursty distribution, malformed
line count 0–100, query, filter values, seed, and time limit. The Judge
regenerates data, checks exact output, and measures runtime, throughput,
robustness, and available resource data. This is a validation benchmark.
