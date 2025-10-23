# 11 - Tiled GEMM 

**Goal:**
Learn how to optimize GEMM with tiling.

**Files:**
- 'main.cpp' - CPU version
- 'main.cu' - GPU version, comparison between naive and tiled versions

**Concepts:**
- The CUDA pipeline: Host init → cudaMalloc → H→D → kernel → sync → D→H → check → free.
- Tiled GEMM(General Matrix Multiplication)