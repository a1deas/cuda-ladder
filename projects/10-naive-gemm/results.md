# Results - 10 - Naive GEMM
**Conditions:**
- Size: 512x512x512 (M, N, K)
- Block: 16x16
- Grid: ((N + 15) / 16, (M + 15) / 16)
- Precision: FP32

**CPU Output:**
```text
CPU: 60.522 ms
FLOP: 4.4 GFLOP/s
```
**GPU Output:**
```text
GPU: 4.016 ms
FLOP: 66.8 GFLOP/s
Max diff: 0.000046(4.6e-5)
```

**Notes:**
- Naive GEMM, as it is in this project Naive FP32 GEMM.
- Even naiveness gives a 15-20x acceleration because of parallel processing.
- Memory and cache are our bottleneck for now. 
