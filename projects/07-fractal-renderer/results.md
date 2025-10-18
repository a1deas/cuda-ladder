# Results - 07 - Fractal nRederer
**Conditions:**
- WidthXHeight: 1920x1080 
- Iterations: 500;
- GPU: RTX 5060;

**GPU Output:**
- Mandelbrot set as it should be:
![screenshot](img/fractal.png)

**Performance:**
- Grid Size: [80, 45, 1]
- Block Size: [16, 16, 1]
- GPU Time: 344,16ms
- Achieved Occupancy: 74.8%
- Warp Exec. Efficiency: 97.9%
- Branch Efficiency: 99.4%
- DRAM Throughput: 184 GB/s
- L2 Hit Rate: 52.6%
- Global Load Efficiency: 91.2%
- Global Store Efficiency: 95.8%

**Summary:**
- Bound Type: **Memory-bound**
- Arithmetic Intensity: 0.24 FLOPs/byte
- Optimizations: coalesced access, stable divergence, shared memory unused

**Notes:**
- 1 thread = 1 pixel.
- Coalesced uchar4 write.
- Limiter — DRAM bandwidth (≈ 60 % of peak).
- Possible improvements: reuse values via shared memory or iterative tiling