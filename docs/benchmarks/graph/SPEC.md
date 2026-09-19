# Graph-processing CLI benchmark

The candidate is a TinyCC-built C program. It reads one graph and its queries
from standard input and writes exactly one LF-terminated result per query.
Successful processing exits 0. The input begins with
`vertices edges directed weighted queries`. Then follow `edges` lines
`u v weight`, then `queries` lines `operation u v`. Vertex identifiers range
from 0 to `vertices-1`. Weights are positive integers. An unweighted graph
still supplies weight 1. An undirected edge applies in both directions.

Public operations:

| Operation | Output |
| --- | --- |
| `reach u v` | `1` if a path exists from `u` to `v`, otherwise `0`. |
| `distance u v` | Minimum total edge weight, or `-1` if unreachable. |
| `neighbors u v` | Distinct outgoing neighbors of `u`, ascending and comma-separated, or `-` if none; `v` is ignored. |
| `components u v` | Number of connected components in an undirected graph; `u` and `v` are ignored. |

Validated challenges choose 1–200 vertices, edge density 0–100%, one of the
public topology distributions (`random`, `path`, `cycle`, `star`,
`disconnected`), directed/undirected and weighted/unweighted modes, a
nonempty operation subset, 1–100 queries, a 32-bit seed, and a per-case time
limit. `components` is available only for undirected graphs. The Judge
regenerates graph and queries from the recipe and seed, checks exact output,
and measures median runtime, input size, largest passing graph, exit/timeout
status, and available resource data. Correctness precedes performance reward.
The candidate runs in the offline Docker sandbox.
