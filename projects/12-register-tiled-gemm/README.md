# 12 - Register Tiled GEMM 

**Goal:**
Learn how to optimize GEMM with register tiling.

**Files:**
- 'main.cpp' - CPU version
- 'main.cu' - GPU version, comparison between naive, tiled and register tiled versions

**Concepts:**
- The CUDA pipeline: Host init → cudaMalloc → H→D → kernel → sync → D→H → check → free.
- Register-Tiled GEMM(General Matrix Multiplication).