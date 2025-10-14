# 05 - Shared Memory Tiling

**Goal:**
Understand how to use **shared memory** to increase memory efficiency and eliminate non-coalescing accesses. 

**Files:**
- 'main.cu' - Naive and Tiled transpose implementations.

**Concepts:**
- CUDA pipeline: **Host init → cudaMalloc → H→D copy → kernel launch → sync → D→H copy → check → free**  
- Shared memory — extremely fast on-chip memory (≈100× faster than global).  
- Naive vs. Tiled transpose.
- Tiled transpose helps to **avoid non-coalesced memory access** and achieve higher throughput.
