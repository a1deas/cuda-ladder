# Results - 12 - Register Tiled GEMM
**Conditions:**
- Size: 512x512x512 (M, N, K)
- Block: 16x16
- Grid: 32x32
- TILE: 32
- Register Tile(REG_TILE): 2x2
- Precision: FP32

**CPU Output:**
```text
CPU Time: 126.551 ms
FLOP: 2.1 GFLOP/s
```

**GPU Naive GEMM Output:**
```text
Naive GEMM: 3.961 ms
FLOP: 67.8 GFLOP/s
Max Diff: 0.000046(4.6e-5)
```

**GPU Tiled GEMM Output:**
```text 
Tiled GEMM: 3.087 ms
FLOP: 87.0 GFLOP/s
Max Diff: 0.000046(4.6e-5)
```

**GPU Register Tiled GEMM Output:**
```text 
Register Tiled GEMM: 1.290 ms
FLOP: 208.0 GFLOP/s
Max Diff: 0.000046(4.6e-5)
```

**Notes:**
- Register tiling allows each thread to compute a 2×2 block of C, reusing loaded values in registers and reducing shared-memory reads.
- As a result: +2.4x on Tiled and 208 GFLOP/s on 512³.
- Max Diff remains stable — everything as it should be.
- Every thread processess more math per one shared fetch(2x2 block —> 4 FMA on aReg/bReg).
- Shared memory traffic per thread decreases, while data reuse and arithmetic intensity (FLOP/byte) increase.
- **Bound-type**: is closer to compute-bound.