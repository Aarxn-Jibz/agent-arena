# Search/indexing validation benchmark

The candidate is a TinyCC-built C program with two modes:

- `solution build`: reads document count, then one space-separated document
  per line. It writes an opaque binary index of its own choosing.
- `solution query <exact|prefix>`: reads four big-endian bytes giving the index
  length, then those index bytes, then one LF-terminated query word. It writes
  ascending matching zero-based document IDs separated by single spaces and
  terminated by LF. No match is a lone LF.

Matching is case-sensitive on whole ASCII words. `exact` matches equal words;
`prefix` matches words beginning with the query. Repeated occurrences in a
document count once. The candidate chooses its index representation and may
rebuild or rewrite it freely.

Challenges vary 0–1,000 documents, 1–100 words per document, 1–30 queries,
uniform/skewed word distributions, matching mode, seed, and time limit. The
Judge generates documents and queries deterministically, checks exact results,
and measures indexing time, median query time, index bytes, correctness, and
available resource status. This is a validation benchmark.
