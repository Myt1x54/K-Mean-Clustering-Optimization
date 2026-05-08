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
POINTS="${POINTS:-100000}"
CLUSTERS="${CLUSTERS:-8}"
MAX_ITERS="${MAX_ITERS:-100}"
MIN_COORD="${MIN_COORD:-0}"
MAX_COORD="${MAX_COORD:-1000}"
THRESHOLD="${THRESHOLD:-1e-4}"
SEED="${SEED:-42}"
REPEATS="${REPEATS:-1}"

threads=(1 2 4 8 16)

if [[ "$REPEATS" -lt 1 ]]; then
    echo "REPEATS must be >= 1"
    exit 1
fi

parse_total_runtime() {
    awk '/Total runtime \(ms\):/ { value = $NF } END { if (value != "") print value }'
}

mean_from_stdin() {
    awk '{ sum += $1; count += 1 } END { if (count > 0) printf "%.6f", sum / count; else print "0.000000" }'
}

stdev_from_stdin() {
    local mean_value="$1"
    awk -v mean="$mean_value" '{ diff = $1 - mean; sumsq += diff * diff; count += 1 } END { if (count > 1) printf "%.6f", sqrt(sumsq / (count - 1)); else print "0.000000" }'
}

run_once() {
    local mode="$1"
    local threads_count="$2"
    "$EXEC" "$mode" "$POINTS" "$CLUSTERS" "$MAX_ITERS" "$MIN_COORD" "$MAX_COORD" "$THRESHOLD" "$SEED" "$threads_count" 2>&1 | parse_total_runtime
}

printf "implementation,threads,points,clusters,repeats,mean_time_ms,stdev_time_ms,speedup,efficiency\n" > "$OUT_CSV"

for t in "${threads[@]}"; do
    echo "Running scaling test for threads=$t"

    seq_times=()
    naive_times=()
    opt_times=()

    for ((rep = 1; rep <= REPEATS; ++rep)); do
        seq_time="$(run_once sequential 1)"
        naive_time="$(run_once parallel "$t")"
        opt_time="$(run_once optimized "$t")"

        if [[ -z "$seq_time" || -z "$naive_time" || -z "$opt_time" ]]; then
            echo "Failed to parse timing output for threads=$t repetition=$rep"
            exit 1
        fi

        seq_times+=("$seq_time")
        naive_times+=("$naive_time")
        opt_times+=("$opt_time")
    done

    seq_mean=$(printf '%s\n' "${seq_times[@]}" | mean_from_stdin)
    naive_mean=$(printf '%s\n' "${naive_times[@]}" | mean_from_stdin)
    opt_mean=$(printf '%s\n' "${opt_times[@]}" | mean_from_stdin)

    seq_stdev=$(printf '%s\n' "${seq_times[@]}" | stdev_from_stdin "$seq_mean")
    naive_stdev=$(printf '%s\n' "${naive_times[@]}" | stdev_from_stdin "$naive_mean")
    opt_stdev=$(printf '%s\n' "${opt_times[@]}" | stdev_from_stdin "$opt_mean")

    for impl in "sequential:1:$seq_mean:$seq_stdev" "naive:$t:$naive_mean:$naive_stdev" "optimized:$t:$opt_mean:$opt_stdev"; do
        IFS=':' read -r implname implthreads implmean implstdev <<< "$impl"
        speedup=$(awk -v s="$seq_mean" -v p="$implmean" 'BEGIN{if (p > 0) printf "%.6f", s/p; else print "0.000000"}')
        efficiency=$(awk -v sp="$speedup" -v th="$implthreads" 'BEGIN{if (th > 0) printf "%.6f", sp/th; else print "0.000000"}')
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$implname" "$implthreads" "$POINTS" "$CLUSTERS" "$REPEATS" "$implmean" "$implstdev" "$speedup" "$efficiency" >> "$OUT_CSV"
    done

done

echo "Wrote: $OUT_CSV"
