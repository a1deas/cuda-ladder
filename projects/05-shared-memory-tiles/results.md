# Results - 05 - Shared Memory Tiles
**Conditions:**
- N = 4096
- Tile size = 32
- Threads = 32

**GPU Output:**
```text
Naive              |   1.458 ms |   92.03 GB/s
Tiled              |   0.884 ms |  151.82 GB/s
```

**Notes:**
- Shared memory size = ~48-100 KB per block(hardware dependent).
- The tiled transpose approach divides the matrix into small tiles, loads each tile into shared memory, performs transpose locally(fast) and then writes the result back to the global memory with coalesced accesses.
- **Tiled** version achieves ~1.65× speedup compared to the **Naive** version.
