# Final Performance Analysis Summary

## Naive OpenMP
- Naive runs show weak scaling due to synchronization overhead from shared updates and lock contention.
- Best observed naive speedup is 1.192, indicating limited parallel efficiency at higher thread counts.

## Optimized OpenMP (Thread-Local Reduction)
- Thread-local accumulation reduces synchronization pressure and improves scalability relative to naive OpenMP.
- Best observed optimized speedup is 1.817, reflecting better parallel utilization.

## SoA Implementation
- SoA layout improves memory locality by keeping hot coordinate streams contiguous.
- Best observed SoA speedup is 4.544.
- Mean IPC: optimized=0.258, soa=0.414. Mean cache miss rate: optimized=0.0045, soa=0.0081.
- Arithmetic intensity trend: optimized=1.8750, soa=2.5000.

## Hardware Bottlenecks
- Scaling plateaus are consistent with memory bandwidth pressure and shared-cache contention at higher thread counts.
- Efficiency drops at larger thread counts indicate parallel overhead and memory subsystem saturation.
- Roofline positioning can be used to explain whether each implementation is closer to memory-bound or compute-bound limits.
