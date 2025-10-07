# 01 - Vector Add 

**Goal:**
First step. 
The purpose is: 
- to understand host↔device memcpy;
- grid/block configuration, kernels, synchronization;
- to check if everything is correct.

**Files:**
- 'main.cpp' - CPU version
- 'main.cu' - GPU version

**Concepts:**
- host↔device memcpy
- cudaEvents(start, stop, elapsed time)
- 'base start' threads = 256, blocks = (N + threads - 1) / threads
- pipeline