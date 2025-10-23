# Results - 11 - Tiled GEMM
**Conditions:**
- Size: 512x512x512 (M, N, K)
- Block: 16x16
- Grid: ((N + 15) / 16, (M + 15) / 16)
- Precision: FP32

**CPU Output:**
```text
CPU: 63.998 ms
FLOP: 4.2 GFLOP/s
```
**GPU Naive GEMM Output:**
```text
GPU: 4.004 ms
FLOP: 67.0 GFLOP/s
Max diff: 0.000046(4.6e-5)
```

**GPU Tiled GEMM Output:**
```text 
GPU: 3.042 ms
FLOP: 88.3 GFLOP/s
Max diff: 0.000046(4.6e-5)
```

**Notes:**
- Shared-memory tiling(16x16) gives ~1.3x vs Naive because task is simple and cache helps naive kernel.
- Accesses to A and B inside tile are coalesced.
- Access to C always after all tiles.
- Bottleneck partly moves from memory to calculations.
