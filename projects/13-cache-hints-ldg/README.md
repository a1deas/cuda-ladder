# 13 - Cache Hints/LDG

**Goal:**
Learn how to use read-only cache via __ldg() and compiler hints (const, __restrict__) to improve memory throughput.

**Files:**
- there is no need in cpp CPU version.
- 'main.cu' - GPU version, comparison between normal read and ldg read.

**Concepts:**
- The CUDA pipeline: Host init → cudaMalloc → H→D → kernel → sync → D→H → check → free.
- GPU Cache
- Memory Hierarchy
- __ldg()
- `__restrict__`