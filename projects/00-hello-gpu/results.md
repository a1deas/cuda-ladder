# Results - 00 - Hello GPU

**CPU Output:**
Hello, CPU!

**GPU Output:**
Hello from Block 2, Thread0
Hello from Block 2, Thread1
Hello from Block 2, Thread2
Hello from Block 0, Thread0
Hello from Block 0, Thread1
Hello from Block 0, Thread2
Hello from Block 1, Thread0
Hello from Block 1, Thread1
Hello from Block 1, Thread2

**Notes:**
- Threads run in parallel, so order of messages is not deterministic.
- 'kernel_hello<<<3, 3>>>' launches 3 blocks with 3 threads each(9 in total)
- Everything executed correctly - Cuda environment workds.