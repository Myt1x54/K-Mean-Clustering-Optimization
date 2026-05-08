# K-Means Sequential Baseline (C++17)

This project provides a clean, modular, and performance-aware **sequential baseline** implementation of the K-Means clustering algorithm for Parallel and Distributed Computing (PDC) research.

The code is intentionally structured so future phases can add:
- OpenMP parallel loops
- thread-local accumulators and synchronization strategies
- scheduling experiments
- cache optimization studies
- profiling and benchmarking hooks

## Project Structure

```text
kmeans_project/
├── CMakeLists.txt
├── README.md
├── benchmark/
│   └── results/
├── scripts/
│   ├── benchmark_baseline.sh
│   └── benchmark_quick.sh
├── include/
│   ├── Point.h
│   ├── Cluster.h
│   ├── KMeans.h
│   ├── Timer.h
│   └── Utils.h
├── src/
│   ├── main.cpp
│   ├── Point.cpp
│   ├── Cluster.cpp
│   ├── KMeans.cpp
│   ├── Timer.cpp
│   └── Utils.cpp
└── data/
```

## Build Instructions (Linux)

```bash
cd kmeans_project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Executable:

```bash
./build/kmeans
```

## Run Instructions

Default run:

```bash
./build/kmeans
```

Custom run (minimum required parameters):

```bash
./build/kmeans 100000 20 100
```

Meaning:
- `100000` -> number of points
- `20` -> number of clusters
- `100` -> maximum iterations

Optional extended arguments:

```bash
./build/kmeans <points> <clusters> <max_iters> <min_coord> <max_coord> <threshold> <seed>
```

## Algorithm Overview

1. Generate `N` random 2D points using `std::mt19937` and uniform real distribution.
2. Initialize `K` centroids by sampling from generated points.
3. Repeat until convergence or `maxIterations`:
   - assign each point to the nearest centroid
   - accumulate per-cluster sums and counts
   - update centroid coordinates from accumulators
   - compute centroid movement and check threshold-based convergence
4. Print runtime and convergence statistics.

Euclidean distance used:

```text
d = sqrt((x1 - x2)^2 + (y1 - y2)^2)
```

## Performance Notes

Sequential baseline includes:
- contiguous storage with `std::vector`
- cache-friendly linear loops
- no per-point dynamic allocations in hot path
- explicit accumulator reset/update phases
- code layout friendly to future OpenMP parallelization of the point-assignment loop

## Sample Commands

```bash
./build/kmeans 10000 8 50
./build/kmeans 100000 20 100
./build/kmeans 1000000 32 200 0 5000 1e-5 123
```

## Deterministic Baseline Benchmarking

Benchmark scripts are provided to generate repeatable baseline tables before OpenMP phases.

Run full deterministic matrix (N sweep, K sweep, max-iteration sweep):

```bash
./scripts/benchmark_baseline.sh
```

Generated artifacts:
- `benchmark/results/baseline_results.csv`
- `benchmark/results/baseline_results.md`

Run quick benchmark smoke suite:

```bash
./scripts/benchmark_quick.sh
```

Run the thread-scaling harness with optional repeated trials and mean/stdev aggregation:

```bash
REPEATS=5 POINTS=50000 CLUSTERS=8 MAX_ITERS=80 ./scripts/threads_scaling.sh
```

Supported environment overrides:
- `REPEATS`: number of runs per case, default `1`
- `POINTS`: dataset size, default `100000`
- `CLUSTERS`: number of clusters, default `8`
- `MAX_ITERS`: maximum iterations, default `100`

The CSV output includes:
- `implementation`
- `threads`
- `points`
- `clusters`
- `repeats`
- `mean_time_ms`
- `stdev_time_ms`
- `speedup`
- `efficiency`

Determinism settings used by scripts:
- fixed seed: `42`
- coordinate range: `[0, 1000]`
- convergence threshold: `1e-4`

## OpenMP Parallel (Naive) Integration

This repository includes a naive OpenMP parallel implementation that reproduces the synchronization strategy from the paper "Algoritmo K-Means: versione sequenziale e versione parallela".

Key points:
- The parallel implementation only parallelizes the outer point-assignment loop using `#pragma omp parallel` / `#pragma omp for`.
- Shared cluster accumulators are updated inside `#pragma omp critical` to reproduce the lock contention bottleneck from the paper (no reductions or thread-local accumulators yet).
- Static scheduling with a chunk size of `1000` is used by default; the chunk size can be configured via command-line.

Build with OpenMP enabled (CMake will link OpenMP if available):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run modes:
- `sequential`: run the sequential implementation
- `parallel`: run the naive OpenMP implementation
- `both`: run sequential then parallel and report speedup/efficiency

Examples:

```bash
# sequential with defaults
./build/kmeans sequential 100000 20 100

# parallel with 8 threads
./build/kmeans parallel 100000 20 100 0 1000 1e-4 42 8

# run both sequential and parallel for direct comparison
./build/kmeans both 100000 20 100 0 1000 1e-4 42 8
```

Why the critical section is intentional:
- The paper describes a naive approach that updates shared centroids under a global lock, which creates a synchronization bottleneck as thread count increases.
- We intentionally preserve that behavior to provide a clear baseline for later optimizations (thread-local accumulators, reductions, scheduling experiments).

