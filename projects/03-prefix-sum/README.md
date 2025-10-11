# 03 - Prefix-sum/Scan 

**Goal:**
Practice the GPU scan(prefix-sum) pattern and build a multi-block exclusive scan.

**Files:**
- 'main.cpp' - CPU version
- 'main.cu' - GPU version

**Concepts:**
- The CUDA pipeline: **Host init → cudaMalloc → H→D → kernel → sync → D→H → check → free**.
- Bank-conflict padding — shared array is indexed with a “hole” every 32 ints to avoid conflicts.
- `extern __shared__`
- `__syncthreads()` usage as barrier for threads inside one block.