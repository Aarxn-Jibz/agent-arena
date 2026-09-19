# Local HTTP server benchmark

The candidate is a TinyCC-built C program. The Judge starts it with one
argument: an absolute Unix-domain socket path inside the disposable sandbox.
The candidate listens there and serves HTTP/1.1 requests. Each request uses a
new connection and `Connection: close`; the candidate closes the connection
after its response. There is no general network access.

The public route vocabulary is:

| Request | Required response |
| --- | --- |
| `GET /health` | Status 200, body `ok` followed by LF, `Content-Type: text/plain`. |
| `POST /echo` | Status 200, body equal to the request body byte-for-byte, `Content-Type: application/octet-stream`. |
| `GET /missing` | Status 404, body `not found` followed by LF, `Content-Type: text/plain`. |
| Malformed or partial request, when selected | Status 400, body `bad request` followed by LF, `Content-Type: text/plain`. |

Every response includes an accurate decimal `Content-Length`. Other headers
may be present. Challenge parameters select a nonempty route subset, 1–20
requests, body size 0–512 bytes, concurrency 1–8, whether malformed and partial
requests occur, a seed, and a per-workload time limit. The Judge sends requests
through the isolated Unix socket, checks protocol and body semantics, and
measures throughput, median and p95 latency, exit/timeout status, and resource
status. Workload generation is seeded and repeatable, though wall-clock timing
has normal local variation. Correctness precedes performance reward.

The candidate has no Internet access or host file mounts and may not invoke
external utilities or libraries. This specification leaves its internal design
entirely to the Solver.
