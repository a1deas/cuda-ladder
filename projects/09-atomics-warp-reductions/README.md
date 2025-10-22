# 09 - Atomics&Warp Reductions 

**Goal:**
Practice the reduction pattern, get used to atomics and learn more about warps

**Files:**
- 'main.cpp' - CPU version
- 'main.cu' - GPU version

**Concepts:**
- The CUDA pipeline: Host init → cudaMalloc → H→D → kernel → sync → D→H → check → free.
- Usage of Atomics functions