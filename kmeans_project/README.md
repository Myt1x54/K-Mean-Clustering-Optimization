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

Determinism settings used by scripts:
- fixed seed: `42`
- coordinate range: `[0, 1000]`
- convergence threshold: `1e-4`
