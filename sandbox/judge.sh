#!/bin/sh
# Only this trusted script runs as container root. Candidate runs as uid 65534.
set -eu
cd /work
tar -xf - -C /work
chmod 0644 solution.c stdin.txt
: > compile.stdout
: > compile.stderr
: > program.stdout
: > program.stderr
echo 0 > program.status
echo 0 > elapsed_ms

set +e
if [ -d src ]; then
    find src -type f -exec chmod 0644 {} +
    c_files=$(find src -type f -name '*.c' | sort)
else
    c_files=
fi
# Candidate paths are validated by the host before archive creation.
(ulimit -f 8192; timeout "$COMPILE_SECONDS" tcc solution.c $c_files -o solution) > compile.stdout 2> compile.stderr
compile_status=$?
set -e
echo "$compile_status" > compile.status
if [ "$compile_status" -eq 0 ]; then
    chmod 0755 solution
    start_ns=$(date +%s%N)
    set +e
    if [ "${1:-}" = "--http" ]; then
        timeout --signal=TERM --kill-after=1s "$RUN_SECONDS" \
            python3 /usr/local/bin/http-driver /work/stdin.txt \
            > program.stdout 2> program.stderr
    else
        (ulimit -f "$OUTPUT_BLOCKS"; timeout --signal=TERM --kill-after=1s "$RUN_SECONDS" \
            setpriv --reuid=65534 --regid=65534 --clear-groups -- /work/solution \
            "$@" < /work/stdin.txt) > program.stdout 2> program.stderr
    fi
    program_status=$?
    set -e
    end_ns=$(date +%s%N)
    echo "$program_status" > program.status
    echo $(((end_ns - start_ns) / 1000000)) > elapsed_ms
fi
head -c "$OUTPUT_BYTES" compile.stdout > compile.stdout.bounded
head -c "$OUTPUT_BYTES" compile.stderr > compile.stderr.bounded
head -c "$OUTPUT_BYTES" program.stdout > program.stdout.bounded
head -c "$OUTPUT_BYTES" program.stderr > program.stderr.bounded
tar -cf - compile.status program.status elapsed_ms \
    compile.stdout.bounded compile.stderr.bounded \
    program.stdout.bounded program.stderr.bounded
