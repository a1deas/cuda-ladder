// project 00 - Hello, GPU - main.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello() {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("Hello, GPU!\n");
}

__global__ void kernel_hello() {
    printf("Hello from Block %d, Thread%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    kernel_hello<<<3, 3>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}