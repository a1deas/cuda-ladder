# 10 - Naive GEMM 

**Goal:**
Learn more about general matrix multiplication(GEMM), understand how matrix multiplication works and what possibilities it gives and find its bottleneck.

**Files:**
- 'main.cpp' - CPU version
- 'main.cu' - GPU version

**Concepts:**
- The CUDA pipeline: Host init → cudaMalloc → H→D → kernel → sync → D→H → check → free.
- GEMM(General Matrix Multiplication)
- FP32