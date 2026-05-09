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

## OpenMP Optimized (Thread-Local Accumulators)

The optimized OpenMP implementation removes the critical section bottleneck using thread-local accumulators and parallel reduction:

Key features:
- Pre-allocate flat arrays: `[thread_id * K + cluster_id]` for sumX, sumY, count
- Each thread accumulates into its own region during parallel point-assignment
- Sequential reduction phase merges thread-local accumulators into global clusters
- Eliminates lock contention entirely

Run optimized implementation:

```bash
# Basic optimized run (default static scheduling)
./build/kmeans optimized 100000 20 100 0 1000 1e-4 42 8

# Optimized with dynamic scheduling
./build/kmeans optimized dynamic 100000 20 100 0 1000 1e-4 42 8

# Optimized with guided scheduling
./build/kmeans optimized guided 100000 20 100 0 1000 1e-4 42 8 1000
```

Argument structure: `./build/kmeans optimized [schedule_policy] <points> <clusters> <max_iters> <min> <max> <threshold> <seed> <threads> [chunk_size]`

Supported scheduling policies:
- `static`: pre-distribute iterations evenly (good for balanced workloads)
- `dynamic`: steal-based scheduling (good for unbalanced workloads)
- `guided`: hybrid strategy with exponentially decreasing chunk sizes

## Benchmark Automation Framework

A comprehensive benchmarking system automates performance evaluation across multiple configurations.

### Running the Full Benchmark Suite

Execute the automated benchmark suite:

```bash
./build/kmeans benchmark
```

This runs:
- **420+ experiment combinations** across:
  - Thread counts: 1, 2, 4, 8, 16
  - Dataset sizes: 100k, 500k, 1M points
  - Cluster counts: 5, 10, 20, 50
  - Implementations: sequential, naive, optimized
  - Scheduling policies: static, dynamic, guided
  - Chunk sizes: 100, 1000

- **Multi-run statistics** (5 repetitions per configuration) to compute:
  - Mean runtime
  - Sample standard deviation
  - Speedup (sequential baseline / parallel time)
  - Efficiency (speedup / thread count)

- **Correctness validation** against sequential baseline before collecting results

- **CSV export** with full metrics

### Benchmark Output

The framework produces:

**Console output:** Progress display

```
[22/120] optimized | guided | Threads=8 | Points=1000000 | Clusters=10 | Chunk=1000
[23/120] optimized | static | Threads=8 | Points=1000000 | Clusters=10 | Chunk=100
...
```

**CSV output:** `benchmark/results/benchmark_results.csv`

Columns:
- `implementation`: sequential, naive, optimized
- `schedule`: static, dynamic, guided (N/A for sequential and naive)
- `chunk_size`: OpenMP chunk size (1000, 10000 etc.)
- `threads`: thread count (1–16)
- `points`: dataset size
- `clusters`: cluster count
- `iterations`: actual iterations to convergence
- `runtime_ms`: individual run time
- `mean_runtime_ms`: average across repetitions
- `stddev_ms`: sample standard deviation
- `speedup`: sequential time / parallel time
- `efficiency`: speedup / thread count

### Example CSV Analysis

Using Python/pandas to analyze results:

```python
import pandas as pd

df = pd.read_csv('benchmark/results/benchmark_results.csv')

# Filter optimized results on 1M points, 16 threads
fast = df[(df['implementation'] == 'optimized') & 
          (df['points'] == 1000000) & 
          (df['threads'] == 16)]

# Group by scheduling policy and compute aggregate stats
summary = fast.groupby(['schedule', 'chunk_size']).agg({
    'mean_runtime_ms': ['min', 'max', 'mean'],
    'speedup': 'mean',
    'efficiency': 'mean'
})

print(summary)
```

### Customizing the Benchmark

To modify the benchmark grid, edit [src/BenchmarkRunner.cpp](src/BenchmarkRunner.cpp) and adjust:

```cpp
std::vector<int> threadCounts = {1, 2, 4, 8, 16};
std::vector<std::size_t> pointCounts = {100000, 500000, 1000000};
std::vector<int> clusterCounts = {5, 10, 20, 50};
std::vector<std::string> schedules = {"static", "dynamic", "guided"};
std::vector<int> chunkSizes = {100, 1000};
int repetitions = 5;
```

Then rebuild:

```bash
cmake --build build -j
```

### Quick Benchmark (Small Configuration)

For rapid iteration, create a custom small-scale run:

```bash
# Edit BenchmarkRunner.cpp to use small grid:
// std::vector<int> threadCounts = {1, 2, 4};
// std::vector<std::size_t> pointCounts = {100000};
// std::vector<int> clusterCounts = {5, 10};
// std::vector<std::string> schedules = {"static"};
// int repetitions = 2;

./build/kmeans benchmark
```

This reduces runtime from hours to minutes while maintaining experimental structure.

### Benchmark Design

The framework:
- **Preserves algorithm implementations** exactly as-is (no modifications to sequential, naive, or optimized logic)
- **Isolates setup time** (data generation, centroid initialization) from timing measurements
- **Includes warm-up run** for each configuration to stabilize cache state
- **Validates correctness** of all implementations using cluster assignment checks
- **Computes speedup/efficiency** against sequential baseline for direct comparison
- **Exports reproducible CSV** compatible with Python/pandas/matplotlib

### Performance Considerations

- Full benchmark suite runs ~420 experiments × 5 repetitions = 2100 individual K-Means executions
- Estimated runtime on typical HPC node: 2–4 hours depending on hardware
- Suitable for overnight batch jobs or cloud computing
- Results enable scalability studies and scheduling policy analysis
- CSV enables histograms, heatmaps, and speedup plots

