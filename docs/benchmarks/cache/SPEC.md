# Cache engine validation benchmark

The candidate is a TinyCC-built C program. Standard input begins with
`capacity operations`, then one command per line: `PUT key value` or
`GET key`. Keys and values are nonnegative decimal integers. It emits one
line per command. `GET` emits `HIT value` or `MISS`. `PUT` on an existing key
updates it and emits `STORED`. `PUT` when space remains stores it and emits
`STORED`. When full, the candidate may choose **any currently stored key** to
remove; it emits `EVICT key` and stores the new key. The Judge follows that
choice and validates every later result. No eviction preference is prescribed.

Validated challenges vary capacity 1–128, 1–1,000 commands, 1–256 possible
keys, a uniform/hot/sequential access distribution, a seed, and a time limit.
The Judge measures state consistency, hit/miss rate, throughput, runtime,
exit/timeout status, and available resource data. This is a validation
benchmark, not a training task selector. Generated C runs only in Docker.
