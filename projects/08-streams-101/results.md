# Results - 08 - Streams 101 - CUDA Streams

## 100M elements -- Low Arithmetic Intensity
**Conditions:**
- 100M elements
- 2 FLOP per element.
- 8 bytes read + 4 bytes write = 12 traffic bytes.
- Intensity = ~0.17 FLOP/byte -> memory-bound task.
- GPU -- 4 streams.

**CPU Output:**
CPU Time: 32.07ms 

**GPU Output:**
GPU Time: 165.003ms

> The results are unexpectedly instructive. GPU is slower than CPU when it goes to read-write without many calculations.

## To be continued with another tests
