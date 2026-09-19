# C language essentials

## Values and types

`char` is one byte; its signedness is implementation-dependent. Use
`unsigned char` when treating bytes as numbers. `size_t` is the unsigned type
used for object sizes and many library byte counts. Integer arithmetic can
overflow: signed overflow is undefined behavior, while unsigned arithmetic
wraps modulo its range. A conversion can narrow a value.

## Pointers and lifetime

A pointer may refer to a live object, one past an array, or null. Dereference
only a live object within bounds. Array indexing `p[i]` requires the same
bounds. Local automatic objects stop existing when their block ends; a pointer
to one must not escape for later use. `malloc` returns owned storage or null;
check before use, and `free` it once. `realloc` can move storage: keep the old
pointer until success is known.

## Strings and bytes

A C string ends at the first zero byte; arbitrary binary data does not.
Carry an explicit length for binary buffers. `strlen` cannot determine an
arbitrary byte stream's length. `memcmp` and `memcpy` use caller-supplied
lengths; verify capacity first. `memmove` handles overlapping ranges.

## Undefined behavior checks

Avoid out-of-bounds access, use-after-free, uninitialized reads, division by
zero, invalid shifts, signed overflow, and mismatched `printf`/`scanf` format
types. Check all sizes before adding or multiplying them for allocation.
Do not assume a successful read fills the requested buffer.
