// project 01 - Vector Add - main.cu
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_OK(stmt) do {                                     \
    cudaError_t err = (stmt);                                  \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s at %s:%d\n",            \
                cudaGetErrorString(err), __FILE__, __LINE__);  \
        return 1;                                              \
    }                                                          \
} while(0)

__global__ void vectorAdd(const int* A, const int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
    
}

int main() {
    cudaEvent_t start, stop;

    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));

    // 20 = estimated 1 million
    // 28 = 268,435,456
    int N = 1 << 28;

    // GPU loves warp multiplicity(32 threads).
    // 256 = 8 warps --> mostly a good choice
    int threads = 256; // optimum
    int blocks = (N + threads - 1) / threads;
    std::vector<int> hA(N), hB(N), hC(N), hRef(N);

    // we should provide device memory
    int *dA=nullptr, *dB=nullptr, *dC=nullptr;
    size_t bytes = N * sizeof(int);
    CUDA_OK(cudaMalloc(&dA, bytes));
    CUDA_OK(cudaMalloc(&dB, bytes));
    CUDA_OK(cudaMalloc(&dC, bytes));
    
    // init arrays
    for (int i = 0; i < N; i++) {
        hA[i] = i;
        hB[i] = 2 * i;
    }

    // copying Host --> Device
    CUDA_OK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    // starting recording
    cudaEventRecord(start);
    // launch kernel
    vectorAdd<<<blocks, threads>>>(dA, dB, dC, N);
    // stoping recording
    cudaEventRecord(stop);

    // check errors + syncronize
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(kerr));
        return 1;
    }
    CUDA_OK(cudaEventSynchronize(stop));

    // copying Device --> Host
    CUDA_OK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0; i < N; i++) {
        hRef[i] = hA[i] + hB[i];
    }
    bool ok = true;
    for (int i = 0; i < 10; i++) {
        if (hC[i] != hRef[i]) { ok = false; break; }
    }
    printf("Correctness (first 10): %s\n", ok ? "OK" : "MISMATCH");

    // result
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU kernel time: %.3f ms\n", ms);

    // deleting events and freeing some space
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    return 0;
}
