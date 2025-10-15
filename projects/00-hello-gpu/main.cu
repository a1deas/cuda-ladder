// project 00 - Hello, GPU - main.cu
#include <cstdio>
#include <cuda_runtime.h>

// Macro to check errors
#define CUDA_OK(stmt) do { \
  cudaError_t err = (stmt); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", \
      cudaGetErrorString(err), __FILE__, __LINE__); \
    return 1; \
  } \
} while(0)

__global__ void hello() {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("Hello, GPU!\n");
}

__global__ void kernel_hello() {
    printf("Hello from Block %d, Thread%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    // Launch kernel
    kernel_hello<<<3, 3>>>();
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    return 0;
}