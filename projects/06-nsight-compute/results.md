# Results - 06 - Nsight Compute

- Computed 02-reduction.
- Computed 03-prefix-sum.

**02-Reduction**
- `atomic` >> `shared` >= `warp`

**03-Prefix-Sum**
- 3 kernels: block Blelloch(first), Blelloch for block sums, addBlockOffsets.
- First is balanced(~77 µs): compute~66%, memory~56%(probably no bottlenecks).
- Block sums(~10 µs): low load (1 block × 1024).
- addBlockOffsets: more memory-bound(~53%).
- CSV: see `binaries/reports/03-prefix-sum.csv`.

**Notes:**
- Nsight Compute(ncu) -- kernel profiler: demonstrates detailed metrics inside kernel(memory, warps, instructions, occupancy, etc).
- Profiling helps to **measure**, not guess why kernel is slow. It gives detailed metrics. 
- Nsight is a must-have for CUDA developer.
