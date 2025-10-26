# Results - 13 - Cache Hints/LDG
**Conditions:**
- N = 1 >> 24; ~16M elements
- threads = 256;
- block = ~62500;
- 

**GPU Output 1:**
```text
Normal: 5.305 ms
LDG: 0.510 ms
Check: 0.200000
```

**GPU Output 1:**
```text
Normal: 0.799 ms
LDG: 0.457 ms
Check: 0.200000
```

**GPU Output 1:**
```text
Normal: 0.888 ms
LDG: 0.535 ms
Check: 0.200000
```

**Notes:**
- First run is slower due to GPU warm-up and cold cache.
- __ldg() means that data is read-only.
- `__restrict__` means that the pointer does not intersect with others.
- const + `__restrict__` let us use cache more effectively.  
- LDG really accelerates read.