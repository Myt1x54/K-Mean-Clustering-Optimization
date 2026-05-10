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

## Memory-Optimized SoA Implementation

This project now includes a memory-optimized implementation that uses a Structure-of-Arrays (SoA) layout to improve memory locality and cache utilization.

Key points:
- The new implementation is provided as a separate class `KMeansSoA` and source `src/KMeansSoA.cpp`.
- Internal layout uses contiguous arrays: `std::vector<double> xs`, `std::vector<double> ys`, and `std::vector<int> clusters`.
- Thread-local accumulators use padded per-thread strides to reduce false sharing (`stride = K + padding`).
- OpenMP scheduling is supported via `omp_set_schedule()` and `schedule(runtime)` so policies (`static`, `dynamic`, `guided`) can be tested at runtime.

How to run the SoA memory-optimized implementation via the benchmark suite:

1. Rebuild:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

2. Run the benchmark suite (includes SoA):

```bash
./build/kmeans benchmark
```

3. The `benchmark` mode includes `soa` in the implementation list so results for SoA will appear in `benchmark/results/benchmark_results.csv` where `implementation` equals `soa`.

Notes on correctness and comparison:
- The SoA implementation is independent from the existing AoS `KMeans` code and does not modify it.
- The benchmark framework validates SoA results before recording them.
- Use the CSV export to compare `soa` against `sequential`, `naive`, and `optimized` rows.

Rationale (AoS vs SoA):
- AoS (Array-of-Structures) is convenient but can hamper vectorized loads and contiguous memory access for hot fields (x and y).
- SoA (Structure-of-Arrays) stores each field contiguously which improves spatial locality when scanning coordinates, and reduces cache misses when processing one field at a time.
- SoA is often a better starting point for future SIMD/vectorization and hardware profiling.

## Profiling with `perf` (Hardware-Level Metrics)

This project now includes a lightweight profiling integration using Linux `perf` to collect hardware-level metrics for detailed bottleneck analysis (cache, instructions, cycles, IPC, synchronization).

Files added:
- `src/ProfileRunner.h` / `src/ProfileRunner.cpp` — C++ helper that runs `perf stat` against the local `kmeans` binary, parses selected metrics, and writes `profiling/profiling_results.csv`.
- `scripts/profile_runner.sh` — simple shell wrapper for ad-hoc profiling runs.

### WSL perf workaround

On WSL, the generic `/usr/bin/perf` wrapper can fail with a kernel-mismatch error even when a working kernel-specific binary exists under `/usr/lib/linux-tools/.../perf`.

This project now resolves a usable binary automatically by searching:
- `PERF_BIN` if set
- `/usr/lib/linux-tools/**/perf`
- fallback generic `perf` only if no kernel-matched binary is found

If you already know the working binary, set it explicitly:

```bash
export PERF_BIN=/usr/lib/linux-tools/5.15.0-177-generic/perf
./build/kmeans profile optimized 8 50000 10 static 1000
```

The helper script also supports the same override:

```bash
PERF_BIN=/usr/lib/linux-tools/5.15.0-177-generic/perf ./scripts/profile_runner.sh optimized 8 50000 10 static 1000
```

Quick usage (ad-hoc):

```bash
# Run optimized implementation under perf (8 threads)
./scripts/profile_runner.sh optimized 8 100000 10 static 1000
```

CLI profiling mode:

You can also run profiling directly via the binary:

```bash
./build/kmeans profile optimized 8 100000 10 static 1000
```

This runs `perf stat -e cache-misses,cache-references,instructions,cycles,task-clock,branches,branch-misses,context-switches,page-faults` and stores raw output under `profiling/<implementation>/` and a CSV summary at `profiling/profiling_results.csv`.

CPU utilization is derived from `task-clock` and elapsed time, so it still appears in the CSV summary even though it is not passed as a raw perf event.

Under the hood, profiling invocations now use `${PERF_BIN} stat ...` rather than hardcoding `perf stat ...`.

CSV columns include:
- `implementation,threads,schedule,points,clusters,runtime_ms,cache_misses,cache_references,cache_miss_rate,instructions,cycles,ipc,cpu_utilization`

Interpretation:
- `ipc = instructions / cycles` indicates instruction throughput — low IPC with high CPU utilization may indicate memory stalls.
- `cache_miss_rate = cache_misses / cache_references` indicates cache pressure.
- Compare `soa` vs `optimized` to observe reduced cache misses and improved IPC for SoA at higher thread counts.

Notes:
- Profiling is optional and separate from normal benchmarking.
- The profiling integration intentionally does not change algorithm code or timing collection.
- Future work: `perf record` flamegraphs, Roofline model helper, LIKWID/VTune adapters (not implemented here).

## Scalability and Roofline Analysis

The project now includes a separate scalability analysis path for final report evaluation. It keeps the benchmark and profiling flows intact while exporting report-ready scalability and Roofline data.

### Run Scalability Experiments

Default scalability run:

```bash
./build/kmeans scalability
```

Select implementations and dataset settings:

```bash
./build/kmeans scalability optimized,soa 50000,100000 10 1000 3
```

Environment overrides:

```bash
SCAL_THREADS=1,2,4,8,16 \
SCAL_SCHEDULES=static,dynamic,guided \
SCAL_CHUNKS=100,1000 \
SCAL_OUTDIR=report \
SCAL_PROFILING_CSV=profiling/profiling_results.csv \
./build/kmeans scalability
```

The runner automatically caps thread counts to the detected hardware limit when requested.

### Generated Outputs

Scalability analysis writes:
- `report/scalability_results.csv`
- `report/roofline_metrics.csv`
- `report/tables/scalability_summary.md`
- `report/tables/roofline_summary.md`
- figures in `report/figures/`

Generate plots from the CSV outputs:

```bash
python3 scripts/generate_scalability_plots.py
```

Generate assets with custom output directory, figure formats, and resolution:

```bash
python3 scripts/generate_scalability_plots.py \
  --input-csv report/scalability_results.csv \
  --outdir report \
  --figures-subdir figures \
  --tables-subdir tables \
  --dpi 350 \
  --formats png,pdf
```

Typical figures:
- `runtime_scaling.png` / `runtime_threads.png`
- `runtime_dataset_size.png`
- `runtime_clusters.png`
- `speedup_scaling.png`
- `speedup_comparison.png`
- `efficiency_scaling.png`
- `efficiency_comparison.png`
- `ipc_scaling.png`
- `cache_behavior.png`
- `scheduling_runtime.png`
- `scheduling_speedup.png`
- `scheduling_comparison.png`
- `scalability_plot.png`
- `roofline_plot.png`
- `roofline_analysis.png`
- `arithmetic_intensity.png`
- `scalability_summary.png`

Presentation-oriented figure set:
- `presentation_scalability_overview.png`

All figures are exported in each requested format (for example `png` and `pdf`) to support both report insertion and slide decks.

### Final Report and Table Exports

The visualization script now writes report-ready summaries in CSV, Markdown, and LaTeX:

CSV summaries:
- `report/final_runtime_summary.csv`
- `report/final_speedup_summary.csv`
- `report/final_efficiency_summary.csv`
- `report/final_hardware_summary.csv`
- `report/final_roofline_summary.csv`
- `report/cache_analysis.csv`
- `report/ipc_analysis.csv`
- `report/comparison_table_presentation.csv`

Markdown/LaTeX tables:
- `report/tables/scalability_summary.md` and `.tex`
- `report/tables/roofline_summary.md` and `.tex`
- `report/tables/hardware_summary.md` and `.tex`
- `report/tables/final_comparison_summary.md` and `.tex`
- `report/tables/analysis_summary.md`

The generated analysis summary highlights:
- naive OpenMP synchronization bottlenecks
- optimized thread-local reduction gains
- SoA locality and IPC behavior
- likely hardware limits (memory bandwidth and cache pressure)

### Formulas

Speedup and efficiency are computed as:

$$
Speedup(T) = \frac{Runtime_{sequential}}{Runtime_T}
$$

$$
Efficiency(T) = \frac{Speedup(T)}{T}
$$

Arithmetic intensity is estimated as:

$$
AI = \frac{FLOPs}{Bytes\ Accessed}
$$

For report analysis, the code exports approximate Roofline-ready metrics using measured runtime, IPC, cache miss rate, achieved GFLOP/s, and an estimated bandwidth proxy. The SoA layout is modeled with better effective locality than AoS, which increases the estimated AI and usually improves achieved performance at higher thread counts.

Runtime, speedup, and efficiency charts are generated across:
- thread counts
- dataset sizes (`points`)
- cluster counts (`clusters`)

Scheduling comparisons are generated across:
- OpenMP schedules (`static`, `dynamic`, `guided`)
- chunk sizes
- thread counts

### Interpretation Guide

- Naive OpenMP is synchronization-bound because `critical` sections serialize the hot path.
- Optimized OpenMP removes the lock bottleneck, so memory traffic and cache pressure become the main limits.
- SoA improves locality and reduces cache waste, which usually raises IPC and improves scalability.
- Scaling eventually plateaus when memory bandwidth, shared-cache pressure, or OpenMP runtime overhead dominates.
- Roofline plots help explain whether a run is compute-bound or memory-bound by comparing achieved performance to the estimated bandwidth and compute ceilings.

### ERT / Roofline Tooling

If an Empirical Roofline Tool is available on your system, you can plug it into the same `report/` workflow later. The current implementation does not require ERT and falls back to a simplified empirical Roofline using the measured profiling data already exported by this project.

### WSL Notes

On WSL, use a kernel-matched perf binary when needed:

```bash
export PERF_BIN=/usr/lib/linux-tools/5.15.0-177-generic/perf
./build/kmeans profile optimized 8 50000 10 static 1000
./build/kmeans scalability
```

The profiling runner now auto-detects `PERF_BIN` and will search `/usr/lib/linux-tools/**/perf` before falling back.

## Visualization Extensibility Notes

The plotting/export pipeline is intentionally modular and additive so future work can extend outputs without changing core K-Means implementations. Planned extension points include:
- SIMD comparison overlays
- GPU kernel result overlays
- MPI/distributed scaling views
- NUMA-aware hardware comparison panels

These extensions are not implemented in this phase.


This SoA variant preserves algorithm logic and correctness while improving memory locality and reducing false sharing via padding.

