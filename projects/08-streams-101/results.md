# Results - 08 - Streams 101 - CUDA Streams

## 100M elements — Light SAXPY Version
**Conditions:**
- 100M elements
- **FLOP**: 2 FLOP per element
- **Read/Write**: 8 bytes read + 4 bytes write = 12 bytes total
- **Intensity** = ~0.17 FLOP/byte -> memory-bound task

**CPU Output:**
CPU Time: 32.07ms 
Validate: OK

**GPU Output:**
Streams: 4
GPU Time: 165.00 ms
Validate: OK

> The results are unexpectedly instructive: **GPU is slower than CPU** when the task is dominated by memory traffic with few computations.

---

## 100M elements — Heavy K-times FMA-sum + SAXPY at the end
**Conditions:**
- 100M elements.
- 100 Iterations.
- **FLOP**: 2 FLOP per iteration x 100 = 200 FLOP/element
- **Read/Write**: 8 bytes read + 4 bytes write = 12 traffic bytes
- **Intersity** = ~16.7 FLOP/byte -> compute-bound task

**CPU Output:**
CPU Time: 7060.27 ms
Validate: OK

**GPU Output:**
Streams: 4
GPU Time: 80.06 ms
Throughput: 249.79 GFLOP/s
PCIe: ~14.99 GB/s
Validate: OK

Streams: 1
GPU Time: 60.70 ms
Throughput: 329.45 GFLOP/s
PCIe: ~19.77 GB/s
Validate: OK

> **1 stream outperforms 4** since in compute-bound mode GPU is fully busy with math and there's little left to overlap — extra streams just add overhead.

---

**Notes:**
- **FLOP(Floating-Point Operation)** — a single arithmetic operation on a floating-point numbers: addition, subtraction, multiplication, division or FMA(fused multiplication + addition). 
- **FLOP/s** — operations per second.
- **GFLOP/s** — billions of operations per second.
- **PCIe(Peripheral Component Interconnect Express)** — the "road" between CPU and GPU. 
    Typical bandwidth: 12-32 GB/s.
    Frequent data transfers can easily negate GPU performance gains.  
- **Intensity(Arithmetic)** — Ratio of computation to transmitted data.
    - "Few FLOPs, many bytes → **memory-bound** (limited by memory or PCIe speed)".
    - "Many FLOPs, few bytes → **compute-bound** (limited by ALU output)".
- **ALU(Arithmetic Logic Unit)** — the basic “calculator” of a processor that performs arithmetic and logic operations. On GPUs, thousands of ALUs work in parallel inside each SM.

