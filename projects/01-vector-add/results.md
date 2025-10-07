# Results - 01 - Vector Add
**Conditions:**
- N = 1 << 28 = 268,435,456 elements
- threads = 256
- blocks = (N + threads - 1) / threads
- correctness checked

**CPU Output:**
CPU Time: 162 ms

**GPU Output:**
GPU kernel time: 12.469 ms

**Notes:**
- N = 1 << 28 takes a lot of memory: ~3GiB just for dA, dB, dC on GPU and ~4GiB for hA, hB, hC, hRef on CPU.
- GPU time throught cudaEventRecort(start/stop) and shows only kernel time(without memcpy H-->D, D-->H).
- Threads: 256 is ok for start, but we could also use 128/512. Probably going to check this later.
- Model: memory-bound. kernel-only BW = ~(12byte/element * N) / time; 
- For N=268,435,456 and 12.469 ms its approximately 240GiB/second. 
- Correctness is ok for both CPU and GPU.