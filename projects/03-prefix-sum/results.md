# Results - 03 - Prefix-sum/Scan
**Conditions:**
- N = 1 << 20;
- threads = 256;
- blocks = ~1954;

**CPU Output:**
CPU Time: 329ms 

**GPU Output:**
GPU Time: 
- KERNEL 1: 173ms;
- KERNEL 2: 0.011ms;
- KERNEL 3: 0.072;

**Notes:**
- `__syncthreads()` is a must-have.
- `extern __shared__ = M * size`. This memory is available for all threads in the block.
- Naive Blelloch in shared by degrees of 2 causes severe memory bank conflicts → this is significantly slower.  
  The classic remedy is **index padding**.  
  Main idea: in shared we have not `M`, but `M + M/NUM_BANKS` elements.
- Multi-block scan (three phases). Per-block scan gives **local** prefixes, so scanning `blockSums` gives **global** prefixes.  
  Main idea is that:  
  - Kernel 1: processes each block locally and prepares block sum  
  - Kernel 2: scans `blockSums` array  
  - Kernel 3: add specific prefixes to each block element

> Common algorithm: load → upsweep (reduce) → exclusive-offset → downsweep


