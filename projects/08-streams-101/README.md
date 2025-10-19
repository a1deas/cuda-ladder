# 08 - Streams 101 - CUDA Streams

**Goal:**
Understand how to use **streams** to increase command efficiency and throughput. 

**Files:**
- 'main.cu' — Light and Heavy SAXPY versions.
- 'main.cpp' — CPU realization.

**Concepts:**
- CUDA pipeline: **Host init → cudaMalloc → H→D copy → kernel launch → sync → D→H copy → check → free**  
- **CUDA Stream** — an ordered sequence of operations on the GPU.
- Overlapping data transfers with calculations and running multiple tasks in parallel.
- Asynchronous execution and synchronization mechanisms.
- **FLOP(Floating-Point Operation)** — basic arithmetic operation.
- **PCIe(Peripheral Component Interconnect Express)** — CPU<->GPU data bus.
- **Arithmetic Intensity** — either memory-bound or compute-bound.
- **ALU(Arithmetic Logic Unit)** — fundamental compute unit inside every **Streaming Multiprocessor**.
- **SM(Streaming Multiprocessor)** — is a small processor inside the GPU.