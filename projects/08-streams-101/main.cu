// project 07 - CUDA Streams(Streams 101) - main.cu
#include <cstdio> 
#include <cstdlib>
#include <vector>
#include <iostream>
#include <chrono>
#include <cassert>
#include <cuda_runtime.h>

// macro to check for the CUDA errors
#define CUDA_OK(stmt) do {                                  \
    cudaError_t err = (stmt);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s at %s:%d\n",         \
            cudaGetErrorString(err), __FILE__, __LINE__);   \
        std::exit(1);                                       \
    }                                                       \
} while(0)

__global__ void saxpyKernel(const float* __restrict__ x,
        float* __restrict__ y,
        float a, 
        int n) 
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float v = y[i];
            for (int k = 0; k < 100; ++k)
                v = a * v + x[i];
            y[i] = v;
        }
}

static void initData(float* x, float* y, int n) {
    for(int i = 0; i < n; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 50000000;
    int numStreams = (argc > 2) ? std::atoi(argv[2]) : 4;
    if (numStreams < 1) numStreams = 1;

    const float a = 3.0f;
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::cout << "N = " << N << " elements, streams = " << numStreams << "\\n";

    float* hX = nullptr;
    float* hY = nullptr;
    CUDA_OK(cudaMallocHost(&hX, bytes));
    CUDA_OK(cudaMallocHost(&hY, bytes));
    initData(hX, hY, N);

    float* dX = nullptr;
    float* dY = nullptr;
    CUDA_OK(cudaMalloc(&dX, bytes));
    CUDA_OK(cudaMalloc(&dY, bytes));

    std::vector<cudaStream_t> streams(numStreams);
    for (int stream = 0; stream < numStreams; ++stream) CUDA_OK(cudaStreamCreate(&streams[stream]));
    cudaEvent_t startAll, stopAll;
    CUDA_OK(cudaEventCreate(&startAll));
    CUDA_OK(cudaEventCreate(&stopAll));

    const int chunkSize = (N + numStreams - 1) / numStreams;
    const int block = 256;

    CUDA_OK(cudaEventRecord(startAll));
    for (int stream = 0; stream < numStreams; ++stream) {
        int offset = stream * chunkSize;
        int n = std::min(chunkSize, N - offset);
        if (n <= 0) break;

        size_t bytesN = static_cast<size_t>(n) * sizeof(float);

        CUDA_OK(cudaMemcpyAsync(
            dX + offset, hX + offset, bytesN, cudaMemcpyHostToDevice, streams[stream]));
        CUDA_OK(cudaMemcpyAsync(
            dY + offset, hY + offset, bytesN, cudaMemcpyHostToDevice, streams[stream]));

        int grid = (n + block - 1) / block;
        saxpyKernel<<<grid, block, 0, streams[stream]>>>(dX + offset, dY + offset, a, n);
        CUDA_OK(cudaGetLastError());

        CUDA_OK(cudaMemcpyAsync(hY + offset, dY + offset, bytesN, cudaMemcpyDeviceToHost, streams[stream]));
    }

    for (int stream = 0; stream < numStreams; ++stream) CUDA_OK(cudaStreamSynchronize(streams[stream]));

    CUDA_OK(cudaEventRecord(stopAll));
    CUDA_OK(cudaEventSynchronize(stopAll));

    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, startAll, stopAll));
    std::cout << "GPU Elapsed time: " << ms << " ms\\n";

    bool ok = true;
    for (int i : {0, N/2, N-1}) {
        if (i < 0 || i >= N) continue;
        float expected = a * 1.0f + 2.0f; // 3*1 + 2 = 5
        if (std::abs(hY[i] - expected) > 1e-4f) {
            std::cerr << "Mismatch at " << i << ": " << hY[i] << " != " << expected << "\\n";
            ok = false; break;
        }
    }
    std::cout << "Validate: " << (ok ? "OK" : "FAIL") << "\\n";

    CUDA_OK(cudaEventDestroy(startAll));
    CUDA_OK(cudaEventDestroy(stopAll));
    for (int stream = 0; stream < numStreams; ++stream) CUDA_OK(cudaStreamDestroy(streams[stream]));
    CUDA_OK(cudaFree(dX));
    CUDA_OK(cudaFree(dY));
    CUDA_OK(cudaFreeHost(hX));
    CUDA_OK(cudaFreeHost(hY));
}