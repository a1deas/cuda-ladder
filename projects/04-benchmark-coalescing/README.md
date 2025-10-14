# 04 - Benchmark Coalescing 

**Goal:**
Understand the difference between coalesced and non-coalesced memory access, and practice GPU benchmarking.

**Files:**
- 'main.cpp' - CPU version
- 'main.cu' - GPU version

**Concepts:**
- CUDA pipeline: **Host init → cudaMalloc → H→D copy → kernel launch → sync → D→H copy → check → free**  
- Coalescing and memory transactions  
- Coalesced vs non-coalesced memory access patterns  
- Command streams — asynchronous command queues  
- Warp = 32 threads × 4 bytes per thread = 128 bytes per warp