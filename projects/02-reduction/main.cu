// project 02 - Reduction - main.cu
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

// macro to check for the CUDA errors
#define CUDA_OK(stmt) do {                                  \
    cudaError_t err = (stmt);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s at %s:%d\n",         \
            cudaGetErrorString(err), __FILE__, __LINE__);   \
        return 1;                                           \
    }                                                       \
} while(0)

// Reduction: Atomic Version
__global__ void reduceAtomic(const int* A, unsigned long long* out, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    unsigned long long local = 0ULL;
    for (int idx = i; idx < N; idx += stride) {
        local += (unsigned long long)A[idx];
    }
    atomicAdd(out, local);
}

// Reduction: Shared Version, one atomic per block
__global__ void reduceShared(const int* A, unsigned long long* out, int N) {
    extern __shared__ unsigned long long result[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    unsigned long long local = 0ULL;
    for (int idx = i; idx < N; idx += stride) {
        local += (unsigned long long)A[idx];
    }
    result[tid] = local;
    __syncthreads();
    
    // blockDim.x must be %2
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            result[tid] += result[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, result[0]);
    }
}

// warp reduction, without divergence
__inline__ __device__
unsigned long long _warpReduceSumULL(unsigned long long v) {
    unsigned mask = 0xFFFFFFFFu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

// Reduction: warp version + shared for step between warps
__global__ void reduceWarp(const int* A, unsigned long long* out, int N) {
    extern __shared__ unsigned long long warpSums[];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x;
    int lane = tid & 31; // pos in warp
    int wid = tid >> 5;  // warp number in block

    unsigned long long local = 0ULL;
    for (int idx = i; idx < N; idx += stride) {
        local += (unsigned long long)A[idx];
    }

    local = _warpReduceSumULL(local); // reduction on warp
    if (lane == 0) warpSums[wid] = local;
    __syncthreads();

    if (wid == 0) {
        unsigned long long blockSum =
            (lane < (blockDim.x >> 5)) ? warpSums[lane] : 0ULL;
        blockSum = _warpReduceSumULL(blockSum);
        if (lane == 0) atomicAdd(out, blockSum);
    }
}


int main() {    
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));

    int N = 1 << 20; // 1 million will be enough
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // host data A = 1
    std::vector<int> hA(N, 1);

    // providing memory buffers
    int* dA = nullptr;
    unsigned long long* dOut = nullptr;
    size_t bytes = N * sizeof(int);

    CUDA_OK(cudaMalloc(&dA,   bytes));
    CUDA_OK(cudaMalloc(&dOut, sizeof(unsigned long long)));

    CUDA_OK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));


    // Atomic version
    CUDA_OK(cudaMemset(dOut, 0, sizeof(unsigned long long)));
    CUDA_OK(cudaEventRecord(start));
    reduceAtomic<<<blocks, threads>>>(dA, dOut, N);
    CUDA_OK(cudaEventRecord(stop));
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "atomic: launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    CUDA_OK(cudaEventSynchronize(stop));
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        unsigned long long hOut = 0ULL;
        CUDA_OK(cudaMemcpy(&hOut, dOut, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        printf("atomic: time = %.3f ms | sum = %llu\n", ms, hOut);
    }


    // Shared version
    CUDA_OK(cudaMemset(dOut, 0, sizeof(unsigned long long)));
    CUDA_OK(cudaEventRecord(start));
    reduceShared<<<blocks, threads, threads * sizeof(unsigned long long)>>>(dA, dOut, N);
    CUDA_OK(cudaEventRecord(stop));
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "shared: launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    CUDA_OK(cudaEventSynchronize(stop));
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        unsigned long long hOut = 0ULL;
        CUDA_OK(cudaMemcpy(&hOut, dOut, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        printf("shared: time = %.3f ms | sum = %llu\n", ms, hOut);
    }


    // Warp version
    CUDA_OK(cudaMemset(dOut, 0, sizeof(unsigned long long)));
    CUDA_OK(cudaEventRecord(start));
    reduceWarp<<<blocks, threads, (threads/32) * sizeof(unsigned long long)>>>(dA, dOut, N);
    CUDA_OK(cudaEventRecord(stop));
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "warp: launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    CUDA_OK(cudaEventSynchronize(stop));
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        unsigned long long hOut = 0ULL;
        CUDA_OK(cudaMemcpy(&hOut, dOut, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        printf("warp:   time = %.3f ms | sum = %llu\n", ms, hOut);
    }

    // free and delete events
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    cudaFree(dA); // 7. cudaFree
    cudaFree(dOut);
    return 0;
}