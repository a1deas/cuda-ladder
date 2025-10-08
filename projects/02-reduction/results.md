# Results - 02 - Reduction
**Conditions:**
- N = 1 << 20 = ~1M elements
- threads = 256
- blocks = (N + threads - 1) / threads
- correctness checked(GPU sum == CPU ref)

**CPU Output:**
CPU Time: 0.423 ms

**GPU Output:**
GPU kernel-only time:
- atomic: 796.828 ms
- shared:   0.045 ms
- warp:     0.028 ms

**Notes:**
- The numbers above are kernel-only. 
- N = 1 << 20 = ~ 4MiB.
- Warp 0.028ms = ~150GB/s
- Shared 0.045ms = ~93GB/s
- Atomic 796.828ms because of contention (many threads updating one counter), not because of DRAM bandwidth.
- CPU 0.423ms, if we add another copies H->D + D->H + warp kernel, end-to-end on GPU will be probably around ~0.33ms, that is still faster than CPU.(Probably will check that later)
- The higher N is, the stronger the GPU wins in terms of core performance.
- To keep atomicAdd, had the accumulator to be unsigned long long and printed with %llu.