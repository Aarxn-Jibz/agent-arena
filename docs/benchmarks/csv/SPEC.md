# CSV/data-processing CLI benchmark

The candidate is a TinyCC-built C program. It receives three arguments:
`<operation> <zero-based-column> <value>`. It reads UTF-8 CSV from standard
input and writes exact UTF-8 output to standard output. A valid input exits 0;
malformed CSV must exit nonzero and emit no standard output. Diagnostics may go
to standard error.

Input has a header `c0,c1,...` and zero or more records. CSV fields may be
quoted. A quoted field may contain commas, doubled quote characters, or
newlines. Output CSV uses the same column order, minimal quoting, doubled
quotes, and LF record endings. There is no extra explanatory text.

Public operation vocabulary:

| Operation | Required result |
| --- | --- |
| `select` | Header and values of the selected column only. |
| `filter_eq` | Header and records whose selected field exactly equals `value`. |
| `sort` | Header and records stably ordered by selected field, comparing UTF-8 byte values. |
| `sum` | Decimal sum of numeric column 0, followed by LF. |
| `uppercase` | Header and all records with ASCII letters in selected field uppercased. |
| `count` | Number of data records as decimal text, followed by LF. |

Validated challenges choose an operation, 0–10,000 rows, 2–8 columns, a
selected column, a short ASCII `value`, a distribution (`plain`, `quoted`,
`mixed`, `wide`), whether to include malformed trailing input, a seed, 1–3
repetitions, and a 1–10,000 ms per-case limit. `sum` uses column 0. The
Challenger can vary these dimensions freely within the schema.

The Judge regenerates cases from the recipe and seed, computes exact expected
output independently, and measures correctness, median runtime, input/output
bytes, the largest passing row count, exit/timeout status, and the time ceiling.
Memory is recorded only if a reliable measurement is available. Correctness
precedes performance reward. The candidate runs in the offline Docker sandbox;
no external library, utility, network, or host data is available.
