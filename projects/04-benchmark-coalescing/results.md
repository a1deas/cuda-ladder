# Results - 04 - Memory Coalescing
**Conditions:**
- N = 1 << 24 = 16,777,216
- threads = 256
- blocks = 65,536

**CPU Output:**
```text
CPU stride( 1) :    5.935 ms |   22.61 GB/s
CPU stride( 2) :    4.437 ms |   15.12 GB/s
CPU stride( 4) :    3.892 ms |    8.62 GB/s
CPU stride( 8) :    3.699 ms |    4.54 GB/s
CPU stride(16) :    3.678 ms |    2.28 GB/s
CPU stride(32) :    2.751 ms |    1.52 GB/s 
```
**GPU Output:**
```text
Coalesced                        |    2.412 ms |   55.65 GB/s
Non-Coalesced (stride = 1)       |    2.359 ms |   56.91 GB/s
Non-Coalesced (stride = 2)       |    2.553 ms |   52.57 GB/s
Non-Coalesced (stride = 4)       |    4.367 ms |   30.73 GB/s
Non-Coalesced (stride = 8)       |    8.589 ms |   15.63 GB/s
Non-Coalesced (stride = 16)      |    9.696 ms |   13.84 GB/s
Non-Coalesced (stride = 32)      |   12.939 ms |   10.37 GB/s
```
**Notes:**
- *Coalescing* is the merging of memory accesses into one or more optimal memory transactions.  
- **Coalesced:** stride = 1  
- **Partially coalesced:** stride = 2 or 4  
- **Non-coalesced:** stride = 8, 16, or 32  
- Coalesced access provides maximum throughput — approximately **55 GB/s**.  
- The higher the stride, the lower the efficiency (stride = 32 → 10.37 GB/s).  
- On CPU, degradation occurs due to cache misses, but the tendency remains the same.  
- Coalescing is a **must-have optimization technique** for achieving optimal GPU performance when accessing global memory.


