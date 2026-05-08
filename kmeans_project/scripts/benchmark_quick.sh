#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTABLE="$ROOT_DIR/build/kmeans"

SEED=42
MIN_COORD=0
MAX_COORD=1000
THRESHOLD=1e-4

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "Executable not found at $EXECUTABLE"
    echo "Run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
    exit 1
fi

cases=(
    "20000 8 80"
    "50000 12 120"
)

for entry in "${cases[@]}"; do
    read -r num_points num_clusters max_iterations <<< "$entry"
    echo "\n=== Quick Benchmark: N=$num_points K=$num_clusters I=$max_iterations ==="
    "$EXECUTABLE" "$num_points" "$num_clusters" "$max_iterations" "$MIN_COORD" "$MAX_COORD" "$THRESHOLD" "$SEED"
done
