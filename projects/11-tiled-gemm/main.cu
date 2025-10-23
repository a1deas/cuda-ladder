// project 11 - Tiled GEMM - main.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// macro to check for the CUDA errors
#define CUDA_OK(stmt) do {                                  \
    cudaError_t err = (stmt);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s at %s:%d\n",         \
            cudaGetErrorString(err), __FILE__, __LINE__);   \
        std::exit(1);                                       \
    }                                                       \
} while(0)

static constexpr int SIZE = 512;
static constexpr int BLOCK_SIZE = 16;

// CPU reference
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

// GPU NAIVE
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

// GPU TILED
__global__ void gemmTiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float AShared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float BShared[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0.0f;
    const int tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < tiles; ++t) {
        int tiledRowA = row;
        int tiledColA = t * BLOCK_SIZE + threadIdx.x;

        int tiledRowB = t * BLOCK_SIZE + threadIdx.y;
        int tiledColB = col;
        
        AShared[threadIdx.y][threadIdx.x] = 
            (tiledRowA < M && tiledColA < K) ? A[tiledRowA * K + tiledColA] : 0.0f;
        
        BShared[threadIdx.y][threadIdx.x] = 
            (tiledRowB < K && tiledColB < N) ? B[tiledRowB * N + tiledColB] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            acc += AShared[threadIdx.y][k] * BShared[k][threadIdx.x];
        }
        __syncthreads();
    }
        if (row < M && col < N)
            C[row * N + col] = acc;
}

int main() {
    const int M = SIZE, N = SIZE, K = SIZE;

    std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);
    for (auto& x : hA) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& x : hB) x = static_cast<float>(rand()) / RAND_MAX;

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    CUDA_OK(cudaMalloc(&dA, sizeA));
    CUDA_OK(cudaMalloc(&dB, sizeB));
    CUDA_OK(cudaMalloc(&dC, sizeC));

    CUDA_OK(cudaMemcpy(dA, hA.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), sizeB, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, end;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&end));

    // CPU baseline for validation
    gemmCPU(hA.data(), hB.data(), hRef.data(), M, N, K);
    
    // warm-up
    gemmNaive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_OK(cudaDeviceSynchronize());

    // NAIVE launch
    CUDA_OK(cudaMemset(dC, 0, sizeC));
    CUDA_OK(cudaEventRecord(start));
    gemmNaive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(end));
    CUDA_OK(cudaEventSynchronize(end));

    float msNaive = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&msNaive, start, end));
    CUDA_OK(cudaMemcpy(hC.data(), dC, sizeC, cudaMemcpyDeviceToHost));

    double maxDiffNaive = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs((double)hC[i] - (double)hRef[i]);
        maxDiffNaive = std::max(maxDiffNaive, diff);
    }
    {
        const double flops = 2.0 * M * N * K;
        const double gflops = flops / (msNaive / 1000.0) / 1e9;
        printf("GPU Naive: %.3f ms, %.1f GFLOP/s\n", msNaive, gflops);
        printf("Max diff: %.6f\n", maxDiffNaive);
    }
    
    // TILED
    CUDA_OK(cudaMemset(dC, 0, sizeC));
    CUDA_OK(cudaEventRecord(start));
    gemmTiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(end));
    CUDA_OK(cudaEventSynchronize(end));

    float msTiled = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&msTiled, start, end));
    CUDA_OK(cudaMemcpy(hC.data(), dC, sizeC, cudaMemcpyDeviceToHost));

    double maxDiffTiled = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs((double)hC[i] - (double)hRef[i]);
        maxDiffTiled = std::max(maxDiffTiled, diff);
    }
    {
        const double flops = 2.0 * M * N * K;
        const double gflops = flops / (msTiled / 1000.0) / 1e9;
        printf("GPU Tiled: %.3f ms, %.1f GFLOP/s\n", msTiled, gflops);
        printf("Max diff: %.6f\n", maxDiffTiled);
    }

    CUDA_OK(cudaFree(dA));
    CUDA_OK(cudaFree(dB));
    CUDA_OK(cudaFree(dC));
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(end));
    return 0;
}