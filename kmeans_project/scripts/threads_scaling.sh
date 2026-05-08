#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXEC="$ROOT_DIR/build/kmeans"
OUT_CSV="$ROOT_DIR/benchmark/results/threads_scaling.csv"

if [[ ! -x "$EXEC" ]]; then
    echo "Executable not found: $EXEC"
    exit 1
fi

# Parameters to run (customize as needed)
POINTS=100000
CLUSTERS=8
MAX_ITERS=100
MIN_COORD=0
MAX_COORD=1000
THRESHOLD=1e-4
SEED=42

threads=(1 2 4 8 16)

printf "threads,points,clusters,sequential_ms,parallel_ms,speedup,efficiency\n" > "$OUT_CSV"

for t in "${threads[@]}"; do
    echo "Running scaling test for threads=$t"

    # Run sequential (single-threaded baseline)
    seq_out=$($EXEC both "$POINTS" "$CLUSTERS" "$MAX_ITERS" "$MIN_COORD" "$MAX_COORD" "$THRESHOLD" "$SEED" "$t" 2>&1)

    # Parse times from both runs (printed by main when mode=both)
    seq_time=$(printf "%s\n" "$seq_out" | awk -F":" '/Sequential time/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')
    par_time=$(printf "%s\n" "$seq_out" | awk -F":" '/Parallel time/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')

    if [[ -z "$seq_time" || -z "$par_time" ]]; then
        echo "Failed to parse timing output for threads=$t"
        echo "$seq_out"
        exit 1
    fi

    speedup=$(awk -v s="$seq_time" -v p="$par_time" 'BEGIN{printf "%.6f", s/p}')
    efficiency=$(awk -v sp="$speedup" -v t="$t" 'BEGIN{printf "%.6f", sp/t}')

    printf "%s,%s,%s,%s,%s,%s,%s\n" "$t" "$POINTS" "$CLUSTERS" "$seq_time" "$par_time" "$speedup" "$efficiency" >> "$OUT_CSV"

done

echo "Wrote: $OUT_CSV"
