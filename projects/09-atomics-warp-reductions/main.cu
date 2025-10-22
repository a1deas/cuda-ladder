// project 09 - Atomics & Warp Reductions - main.cu
#include <iostream>
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

// atomic naive
__global__ void atomicSum(
    const float* __restrict__ data, 
    float* __restrict__ result, 
    int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        atomicAdd(result, data[idx]);
    }
}

// Warp sum
__inline__ __device__
float warpSum(float val) { 
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Warp sum + Atomic
__global__ void blockSum(const float* __restrict__ data, float* __restrict__ result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (idx < N) ? data[idx] : 0.0f;

    float sum = warpSum(x);

    __shared__ float warpSums[32];          // max 32 warps per block
    int lane = threadIdx.x % 32;            // position in warp(0..31)
    int warpId = threadIdx.x / 32;          // warp number in block
    int warpCount = (blockDim.x + 31) >> 5; // 

    if (lane == 0) warpSums[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        float value = (lane < warpCount) ? warpSums[lane] : 0.0f;
        value = warpSum(value);
        if (lane == 0) atomicAdd(result, value);
    }
}

int main() {
    const int N = 100000000;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    float* hData = new float[N];
    for (int i = 0; i < N; i++) hData[i] = 1.0f;
    
    float* dData = nullptr;
    float* dSum = nullptr;
    CUDA_OK(cudaMalloc(&dData, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&dSum, sizeof(float)));
    CUDA_OK(cudaMemcpy(dData, hData, N * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, end;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&end));

    // Naive Atomic
    CUDA_OK(cudaMemset(dSum, 0, sizeof(float)));
    CUDA_OK(cudaEventRecord(start));
    atomicSum<<<GRID, BLOCK>>>(dData, dSum, N);
    CUDA_OK(cudaEventRecord(end));
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "atomic: launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    CUDA_OK(cudaEventSynchronize(end));
    {
        float hSumAtomic = 0.0f;
        float msAtomic = 0.0f;
        CUDA_OK(cudaMemcpy(&hSumAtomic, dSum, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaEventElapsedTime(&msAtomic, start, end));
        printf("Naive Atomic:\nGPU Elapsed time: %.3f ms\nSum = %.1f", msAtomic, hSumAtomic);
    }

    // Warp + Atomic
    CUDA_OK(cudaMemset(dSum, 0, sizeof(float)));
    CUDA_OK(cudaEventRecord(start));
    blockSum<<<GRID, BLOCK>>>(dData, dSum, N);
    CUDA_OK(cudaEventRecord(end));
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "warp+atomic: launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    CUDA_OK(cudaEventSynchronize(end));
    {
        float hSumBlock = 0.0f;
        float msBlock = 0.0f;
        CUDA_OK(cudaMemcpy(&hSumBlock, dSum, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaEventElapsedTime(&msBlock, start, end));
        printf("Warp+Atomic:\nGPU Elapsed time: %f ms\nSum = %f", msBlock, hSumBlock);
    }
    
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(end));
    cudaFree(dSum);
    cudaFree(dData);
    cudaFree(hData);

    CUDA_OK(cudaDeviceReset());
    return 0;
}