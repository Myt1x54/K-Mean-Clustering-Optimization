# Baseline Benchmark Results

Deterministic run settings: seed=42, min_coord=0, max_coord=1000, threshold=1e-4

| Case | N | K | Max Iters | Seed | Total Runtime (ms) | Iterations Executed | Avg Iteration (ms) | Converged |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| N_SWEEP_1 | 10000 | 8 | 100 | 42 | 30.876 | 93 | 0.329 | Yes |
| N_SWEEP_2 | 50000 | 8 | 100 | 42 | 173.565 | 100 | 1.722 | No (hit max iterations) |
| N_SWEEP_3 | 100000 | 8 | 100 | 42 | 325.060 | 100 | 3.231 | No (hit max iterations) |
| K_SWEEP_1 | 100000 | 4 | 100 | 42 | 48.731 | 23 | 2.099 | Yes |
| K_SWEEP_2 | 100000 | 8 | 100 | 42 | 306.223 | 100 | 3.052 | No (hit max iterations) |
| K_SWEEP_3 | 100000 | 16 | 100 | 42 | 452.441 | 74 | 6.090 | Yes |
| ITER_SWEEP_1 | 100000 | 8 | 50 | 42 | 154.571 | 50 | 3.081 | No (hit max iterations) |
| ITER_SWEEP_2 | 100000 | 8 | 100 | 42 | 330.780 | 100 | 3.297 | No (hit max iterations) |
| ITER_SWEEP_3 | 100000 | 8 | 200 | 42 | 542.218 | 174 | 3.106 | Yes |
