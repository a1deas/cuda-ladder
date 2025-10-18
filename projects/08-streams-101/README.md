# 08 - Streams 101 - CUDA Streams

**Goal:**
Understand how to use **streams** to increase command efficiency and throughput. 

**Files:**
- 'main.cu' - GPU Version.
- 'main.cpp' - CPU Version.

**Concepts:**
- CUDA pipeline: **Host init → cudaMalloc → H→D copy → kernel launch → sync → D→H copy → check → free**  
- CUDA Stream - an ordered sequence of operations on the GPU.
- Overlapping copying with calculations and parallelism of multiple tasks.
- Async and synchronizations.