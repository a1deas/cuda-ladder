# 02 - Reduction 

**Goal:**
Practice the reduction pattern (sum of an array) and compare three GPU variants

**Files:**
- 'main.cpp' - CPU version
- 'main.cu' - GPU version

**Concepts:**
- The CUDA pipeline: Host init → cudaMalloc → H→D → kernel → sync → D→H → check → free.
- usage of CUDA functions
- __syncthreads() usage as barrier for threads inside one block
- Why atomics can bottleneck (contention).