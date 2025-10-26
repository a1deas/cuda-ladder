// project 13 - Cache Hints/LDG - main.cu
#include <cuda_runtime.h>
#include <cstdio>

// macro to check for the CUDA errors
#define CUDA_OK(stmt) do {                                  \
    cudaError_t err = (stmt);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s at %s:%d\n",         \
            cudaGetErrorString(err), __FILE__, __LINE__);   \
        std::exit(1);                                       \
    }                                                       \
} while(0)

// Normal read
__global__ void normalRead(const float* A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) 
        B[i] = A[i] * 2.0f;
}

// Read-only cache 
__global__ void ldgRead(const float* __restrict__ A, float* __restrict__ B, int N) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        B[i] = __ldg(&A[i]) * 2.0f;
}

int main() {
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    float* hA = new float[N];
    float* hB = new float[N];

    for (int i = 0; i < N; ++i) hA[i] = i * 0.001f;

    float* dA;
    float* dB;
    CUDA_OK(cudaMalloc(&dA, bytes));
    CUDA_OK(cudaMalloc(&dB, bytes));

    CUDA_OK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Normal
    float ms;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    normalRead<<<grid, block>>>(dA, dB, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    printf("Normal: %.3f ms\n", ms);

    // LDG Read
    cudaEventRecord(start);
    ldgRead<<<grid, block>>>(dA, dB, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    printf("LDG: %.3f ms\n", ms);
    
    CUDA_OK(cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost));

    printf("Check: %f\n", hB[100]);

    cudaFree(dA);
    cudaFree(dB);
    delete[] hA;
    delete[] hB;
}