# 00 - Hello GPU 

**Goal:**
My first CUDA program. 
The purpose is: 
- to test, if the GPU is detected;
- to understand the basic kernel structure;

**Files:**
- 'main.cpp' - CPU version
- 'main.cu' - GPU version

**Concepts:**
- '__global__' kernel
- '<<<blocks, threads>>>' launch syntax
- basic CUDA indexing('blockIdx', 'threadIdx')
- syncronization with 'cudaDeviceSynchronize()'