# Results - 09 - Atomics&Warp Reductions
## First Test - 1M Elements
**Conditions:**
- N = 1M elements
- threads = 256
- blocks = ~3907

**CPU Output:**
- CPU Sum: 1000000.0
- CPU Time: 0.457 ms

**GPU Output:**
Naive Atomic:
- GPU Sum: 1000000.0
- GPU Time: 126.661 ms

Warp+Atomic:
- GPU Sum: 1000000.0
- GPU Time: 0.09 ms

## Second Test - 100M Elements
- N = 100M elements
- threads = 256
- blocks = ~390625

**CPU Output:**
- CPU Sum: 16777216.0
- CPU Time: 43.235 ms

**GPU Output:**
Naive Atomic:
- GPU Sum: 100000000.0
- GPU Time: 119.232 ms

Warp+Atomic:
- GPU Sum: 100000000.0
- GPU Time: 2.068000 ms

**Notes:**
- Naive Atomic — is slow because every atomic is trying to access one memory address.
- Warp + Atomic — every block processess local shared memory reductions and only one atomicAdd per block.
- Performace: 
    - GPU 60x faster on 1M elements.
    - GPU 20-25x faster on 100M elements.
- Bound: Memory-bound.