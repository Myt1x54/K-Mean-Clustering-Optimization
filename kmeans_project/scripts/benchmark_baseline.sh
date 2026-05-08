#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTABLE="$ROOT_DIR/build/kmeans"
RESULTS_DIR="$ROOT_DIR/benchmark/results"
CSV_FILE="$RESULTS_DIR/baseline_results.csv"
MD_FILE="$RESULTS_DIR/baseline_results.md"

SEED=42
MIN_COORD=0
MAX_COORD=1000
THRESHOLD=1e-4

mkdir -p "$RESULTS_DIR"

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "Executable not found at $EXECUTABLE"
    echo "Run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
    exit 1
fi

# Cases are deterministic due to fixed seed and fixed algorithm settings.
cases=(
    "N_SWEEP_1 10000 8 100"
    "N_SWEEP_2 50000 8 100"
    "N_SWEEP_3 100000 8 100"
    "K_SWEEP_1 100000 4 100"
    "K_SWEEP_2 100000 8 100"
    "K_SWEEP_3 100000 16 100"
    "ITER_SWEEP_1 100000 8 50"
    "ITER_SWEEP_2 100000 8 100"
    "ITER_SWEEP_3 100000 8 200"
)

printf "case,num_points,num_clusters,max_iterations,seed,total_runtime_ms,iterations_executed,avg_iteration_ms,converged\n" > "$CSV_FILE"

for entry in "${cases[@]}"; do
    read -r case_name num_points num_clusters max_iterations <<< "$entry"

    echo "Running $case_name (N=$num_points K=$num_clusters I=$max_iterations)"

    output="$($EXECUTABLE "$num_points" "$num_clusters" "$max_iterations" "$MIN_COORD" "$MAX_COORD" "$THRESHOLD" "$SEED")"

    total_runtime_ms="$(printf "%s\n" "$output" | awk -F':' '/Total runtime \(ms\)/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"
    iterations_executed="$(printf "%s\n" "$output" | awk -F':' '/Iterations executed/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"
    avg_iteration_ms="$(printf "%s\n" "$output" | awk -F':' '/Avg iteration \(ms\)/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"
    converged="$(printf "%s\n" "$output" | awk -F':' '/Converged/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$case_name" "$num_points" "$num_clusters" "$max_iterations" "$SEED" \
        "$total_runtime_ms" "$iterations_executed" "$avg_iteration_ms" "$converged" >> "$CSV_FILE"
done

{
    echo "# Baseline Benchmark Results"
    echo
    echo "Deterministic run settings: seed=$SEED, min_coord=$MIN_COORD, max_coord=$MAX_COORD, threshold=$THRESHOLD"
    echo
    echo "| Case | N | K | Max Iters | Seed | Total Runtime (ms) | Iterations Executed | Avg Iteration (ms) | Converged |"
    echo "|---|---:|---:|---:|---:|---:|---:|---:|---|"

    tail -n +2 "$CSV_FILE" | while IFS=',' read -r case_name num_points num_clusters max_iterations seed total_runtime_ms iterations_executed avg_iteration_ms converged; do
        echo "| $case_name | $num_points | $num_clusters | $max_iterations | $seed | $total_runtime_ms | $iterations_executed | $avg_iteration_ms | $converged |"
    done
} > "$MD_FILE"

echo
echo "Wrote: $CSV_FILE"
echo "Wrote: $MD_FILE"
