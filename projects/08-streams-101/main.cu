// project 08 - CUDA Streams(Streams 101) - main.cu
#include <cstdio> 
#include <cstdlib>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
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

// Light version, memory-bound
__global__ void saxpy(const float* __restrict__ x,
        float* __restrict__ y,
        float a, 
        int n) 
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            y[i] = a * x[i] + y[i]; // classical SAXPY
        }

}

// Heavy version K times FMA-sum and SAXPY at the end
__global__ void saxpyHeavy(const float* __restrict__ x, 
        float* __restrict__ y,
        float a,
        int n,
        int iters)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;

            float xIndex = x[i];
            float acc = 0.0f;

            #pragma unroll 4
            for (int k = 0; k < iters; ++k) {
                acc = fmaf(a, xIndex, acc);
            }
            y[i] = acc + y[i];
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
    int iters = (argc > 3) ? std::atoi(argv[3]) : 100;
    if (numStreams < 1) numStreams = 1;

    const float a = 3.0f;
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    printf("N = %i elements, streams = %i, iters = %i\n", N, numStreams, iters);

    // host buffers
    float* hX = nullptr;
    float* hY = nullptr;
    CUDA_OK(cudaMallocHost(&hX, bytes));
    CUDA_OK(cudaMallocHost(&hY, bytes));
    initData(hX, hY, N);

    // device buffers
    float* dX = nullptr;
    float* dY = nullptr;
    CUDA_OK(cudaMalloc(&dX, bytes));
    CUDA_OK(cudaMalloc(&dY, bytes));

    // streams and events
    std::vector<cudaStream_t> streams(numStreams);
    for (int stream = 0; stream < numStreams; ++stream) CUDA_OK(cudaStreamCreate(&streams[stream]));
    cudaEvent_t startAll, stopAll;
    CUDA_OK(cudaEventCreate(&startAll));
    CUDA_OK(cudaEventCreate(&stopAll));

    // break intp chunks
    const int chunkSize = (N + numStreams - 1) / numStreams;
    const int block = 256;

    CUDA_OK(cudaEventRecord(startAll));
    for (int stream = 0; stream < numStreams; ++stream) {
        int offset = stream * chunkSize;
        int n = std::min(chunkSize, N - offset);
        if (n <= 0) break;

        size_t bytesN = static_cast<size_t>(n) * sizeof(float);

        // async copying from host to device
        CUDA_OK(cudaMemcpyAsync(
            dX + offset, hX + offset, bytesN, cudaMemcpyHostToDevice, streams[stream]));
        CUDA_OK(cudaMemcpyAsync(
            dY + offset, hY + offset, bytesN, cudaMemcpyHostToDevice, streams[stream]));

        // kernel launch
        int grid = (n + block - 1) / block;
        saxpyHeavy<<<grid, block, 0, streams[stream]>>>(dX + offset, dY + offset, a, n, iters);
        CUDA_OK(cudaGetLastError());

        // async copying from device to host
        CUDA_OK(cudaMemcpyAsync(hY + offset, dY + offset, bytesN, cudaMemcpyDeviceToHost, streams[stream]));
    }

    // waiting all streams to stop and synchronizing
    for (int stream = 0; stream < numStreams; ++stream) CUDA_OK(cudaStreamSynchronize(streams[stream]));
    CUDA_OK(cudaEventRecord(stopAll));
    CUDA_OK(cudaEventSynchronize(stopAll));

    // print result
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, startAll, stopAll));
    printf("GPU Elapsed time: %f ms\n", ms);

    double timeS = ms / 1e3;
    double gFlops = (double)N * iters * 2.0 / 1e9 / timeS; // 2 FLOP per FMA
    double pcieGbs = (double)N * 12.0 / 1e9 / timeS; // 12 bytes total traffic
    printf("Throughput: %.2f GFLOP/s | PCIe: %.2f GB/s (approx)\n", gFlops, pcieGbs);

    // validation
    bool ok = true;
    float expected = 2.0f + iters * a * 1.0f;
    for (int i : {0, N/2, N-1}) {
        if (i < 0 || i >= N) continue;
        if (std::abs(hY[i] - expected) > 1e-3f) {
            std::cerr << "Mismatch at " << i << ": " << hY[i]
                      << " != " << expected << '\n';
            ok = false; break;
        }
    }
    printf("Validate: %s", (ok ? "OK" : "FAIL")); 

    // destroy events, clean memory
    CUDA_OK(cudaEventDestroy(startAll));
    CUDA_OK(cudaEventDestroy(stopAll));
    for (int stream = 0; stream < numStreams; ++stream) CUDA_OK(cudaStreamDestroy(streams[stream]));
    CUDA_OK(cudaFree(dX));
    CUDA_OK(cudaFree(dY));
    CUDA_OK(cudaFreeHost(hX));
    CUDA_OK(cudaFreeHost(hY));
}