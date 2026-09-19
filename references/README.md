# Local programming references

This frozen pack is original, concise explanatory text for agent lookup. A
future agent should request only the document and section relevant to its
current question. The experiment records each lookup as
`reference_reads: [{"document": "references/c-language.md", "section": "Pointers", "read_at": "..."}]`
in episode evidence. The pack contains language/API facts and arena rules,
not benchmark solutions.

| Document | Topics |
| --- | --- |
| [C language](c-language.md) | Types, pointers, memory, undefined behavior. |
| [C library and binary I/O](c-library.md) | Standard-library calls and exact byte handling. |
| [POSIX](posix.md) | Files, processes, sockets, timing. |
| [TinyCC and arena](tinycc-arena.md) | Compiler use and sandbox restrictions. |

## Provenance and licensing

The prose is newly written for this project from general C/POSIX knowledge and
locally verified arena behavior. It does not reproduce books, manuals, or
third-party reference pages wholesale. Code snippets are short original
examples. Treat this pack as project documentation under the repository's
license; if a formal license is later added, apply it consistently. Recheck
platform-specific claims against the pinned sandbox image before changing the
allowed API surface. Freeze a version or commit hash for each experiment run.
