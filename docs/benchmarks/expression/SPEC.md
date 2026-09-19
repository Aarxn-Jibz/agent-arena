# Expression evaluator benchmark

The candidate is a TinyCC-built C program. It reads one UTF-8 expression line
from standard input. It writes exactly one line to standard output and exits
0: a signed decimal integer followed by LF for a valid expression, or
`ERROR` followed by LF for unsupported syntax, division by zero, or signed
64-bit overflow. Standard error may contain diagnostics.

The public language has decimal integer literals, parentheses, binary `+`,
`-`, `*`, `/`, and optional unary `+` and `-`. Usual precedence and left
associativity apply. Division truncates toward zero. Results must stay in the
signed 64-bit range. Whitespace may separate tokens. Every other syntax is
invalid. Challenges choose a nonempty operator subset; whether unary
operators appear; depth 1–8; token budget 1–60; literal bound 0–1,000,000;
1–20 cases; how many have malformed syntax; a seed; and a per-case time limit.

The Judge generates valid and malformed cases deterministically, computes
exact expected output with an independent restricted oracle, and measures
correctness, malformed-input behavior, exit/timeout status, median runtime,
largest passing input, and resource status. Every case must pass before
performance contributes to reward. Generated C runs only in the offline Docker
sandbox. The specification does not constrain the candidate's internal design.
