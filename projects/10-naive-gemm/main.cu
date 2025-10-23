// project 10 - Naive GEMM - main.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <vector> 
#include <random>

// macro to check for the CUDA errors
#define CUDA_OK(stmt) do {                                  \
    cudaError_t err = (stmt);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s at %s:%d\n",         \
            cudaGetErrorString(err), __FILE__, __LINE__);   \
        std::exit(1);                                       \
    }                                                       \
} while(0)

// CPU
void gemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) { 
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// GPU
__global__ void gemmNaive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) { 
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[ k* N + col];
        C[row * N + col] = sum;
    }
}

int main() { 
    const int M = 512;
    const int N = 512;
    const int K = 512;

    std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    for (auto& x : hA) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& x : hB) x = static_cast<float>(rand()) / RAND_MAX;

    // CPU launch
    gemmCPU(hA.data(), hB.data(), hRef.data(), M, N, K);

    cudaEvent_t start, end;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&end));

    // Device Buffers
    float* dA;
    float* dB;
    float* dC;
    
    CUDA_OK(cudaMalloc(&dA, sizeA));
    CUDA_OK(cudaMalloc(&dB, sizeB));
    CUDA_OK(cudaMalloc(&dC, sizeC));

    CUDA_OK(cudaMemcpy(dA, hA.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), sizeB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    // warm-up
    gemmNaive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaEventRecord(start));

    gemmNaive<<<grid, block>>>(dA, dB, dC, M, N, K);

    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(end));
    CUDA_OK(cudaEventSynchronize(end));

    float ms = 0.0f;

    CUDA_OK(cudaEventElapsedTime(&ms, start, end));
    double flops = 2.0 * M * N * K;
    double gpu_gflops = flops / (ms / 1000.0) / 1e9;
    printf("GPU: %.3f ms, %.1f GFLOP/s\n", ms, gpu_gflops);
    
    CUDA_OK(cudaMemcpy(hC.data(), dC, sizeC, cudaMemcpyDeviceToHost));

    double maxDiff = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs(static_cast<double>(hC[i]) - static_cast<double>(hRef[i]));
        maxDiff = std::max(maxDiff, diff); 
    }
    printf("Max diff: %.6f\n", maxDiff);

    CUDA_OK(cudaFree(dA));
    CUDA_OK(cudaFree(dB));
    CUDA_OK(cudaFree(dC));
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(end));
    return 0;
}