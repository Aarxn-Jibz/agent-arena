# Held-out boss: tiny filesystem simulator

This boss is reserved for a first **zero-shot** baseline-versus-trained
comparison. No adaptation on its results is allowed until both results are
frozen outside training evidence.

The candidate is a TinyCC-built C program invoked as `solution apply` once
per operation, in a fresh disposable container. Standard input begins with a
fixed 4096-byte virtual disk image, followed by one ASCII command and LF.
Standard output must begin with a new 4096-byte image, followed by the exact
response. The Judge passes only that returned image to the next operation.
The initially empty image is all zero bytes. The candidate chooses its own
representation within the image. No state on the host or in the disposable
container may serve as the filesystem implementation.

Paths are `/` followed by 1–12 lowercase letters or digits. At most 32 files
and 2048 total content bytes are supported. Commands and responses:

| Command | Response |
| --- | --- |
| `CREATE path` | `OK` if absent and capacity remains; otherwise `ERR`. |
| `WRITE path hexbytes` | Replace content of an existing file if total content fits; `OK` or `ERR`. |
| `READ path` | `DATA hexbytes` if present; otherwise `ERR`. |
| `DELETE path` | `OK` if present; otherwise `ERR`. |
| `LIST` | Existing paths in ascending order, separated by single spaces; empty line when none. |

Every response ends in LF. Invalid requests receive `ERR` and leave the image
state unchanged. Validated hidden challenges select 1–30 operations, a seed,
and a per-step time limit. The Judge compares each response against a trusted
virtual state, checks image framing, and measures correctness and runtime.
The specification does not constrain allocation, indexing, or recovery design.
