// project 03 - Prefix-sum/Scan - main.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector> 

// macro to check for the CUDA errors
#define CUDA_OK(stmt) do {                                  \
    cudaError_t err = (stmt);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s at %s:%d\n",         \
            cudaGetErrorString(err), __FILE__, __LINE__);   \
        return 1;                                           \
    }                                                       \
} while(0)

// Index Paddings to fix memory banks conflicts
#define LOG_NUM_BANKS 5 // 32 banks
#define NUM_BANKS (1 << LOG_NUM_BANKS)
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)

__global__ void exclusiveBlockBlelloch(const int* __restrict__ in, int* __restrict__ out, int* __restrict__ blockSums, int N) {
    extern __shared__ int s[]; // size M = 2*blockDim.x
    int tid = threadIdx.x;
    int M = 2 * blockDim.x;
    int base = blockIdx.x * M;

    // Init 
    int a = base + 2*tid;
    int b = base + 2*tid + 1;

    // load
    int ia = 2*tid;
    int ib = 2*tid + 1;
    int pa = ia + CONFLICT_FREE_OFFSET(ia);
    int pb = ib + CONFLICT_FREE_OFFSET(ib);
    s[pa] = (a < N) ? in[a] : 0;
    s[pb] = (b < N) ? in[b] : 0;    
    __syncthreads();

    // upsweep
    for (int offset = 1; offset < M; offset <<= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < M) { 
            int p = idx + CONFLICT_FREE_OFFSET(idx);
            int pm = (idx - offset) + CONFLICT_FREE_OFFSET(idx - offset);
            s[p] += s[pm];
        }
        __syncthreads();
    }
    
    // save total 
    if (tid == 0 && blockSums) blockSums[blockIdx.x] = s[(M-1) + CONFLICT_FREE_OFFSET(M-1)];
    if (tid == 0)              s[(M-1) + CONFLICT_FREE_OFFSET(M-1)] = 0;
    __syncthreads();

    // downsweep
    for (int offset = M >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < M) {
            int p   = idx + CONFLICT_FREE_OFFSET(idx);
            int pm  = (idx - offset) + CONFLICT_FREE_OFFSET(idx - offset);
            int t = s[pm];
            s[pm] = s[p];
            s[p] += t;
        }
        __syncthreads();
    }

    // store:
    if (a < N) out[a] = s[pa];
    if (b < N) out[b] = s[pb];
}

__global__ void addBlockOffsets(int* __restrict__ out, const int* __restrict__ blockOffsets, int N) {
    int M = 2 * blockDim.x;
    int base = blockIdx.x * M;
    int tid  = threadIdx.x;
    int a = base + 2*tid;
    int b = base + 2*tid + 1;
    int add = blockOffsets ? blockOffsets[blockIdx.x] : 0;

    if (a < N) out[a] += add;
    if (b < N) out[b] += add;
}

// For testing
void exclusiveScanCPU(const int* in, int* out, int N) {
    int acc = 0;
    for (int i = 0; i < N; ++i) {
        out[i] = acc;
        acc += in[i];
    }
}

int main() {
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));

    const int N = 1 << 20;
    const int threads = 256;
    const int M = 2 * threads;
    const int blocks = (N + M - 1) / M;

    std::vector<int> hIn(N, 1);
    std::vector<int> hOut(N), hRef(N);

    // device buffers
    int *dIn = nullptr; 
    int *dOut = nullptr;
    CUDA_OK(cudaMalloc(&dIn, N * sizeof(int)));
    CUDA_OK(cudaMalloc(&dOut, N * sizeof(int)));
    
    // memory buffers for multiblock
    int *dBlockSums = nullptr;      
    int *dBlockOffsets = nullptr;  
    CUDA_OK(cudaMalloc(&dBlockSums, blocks * sizeof(int)));
    CUDA_OK(cudaMalloc(&dBlockOffsets,blocks * sizeof(int)));
    // copying H->D
    CUDA_OK(cudaMemcpy(dIn, hIn.data(), N*sizeof(int), cudaMemcpyHostToDevice));

    // KERNEL 1
    CUDA_OK(cudaEventRecord(start));
    size_t sharedMemory1 = (M + M/NUM_BANKS) * sizeof(int);
    exclusiveBlockBlelloch<<<blocks, threads, sharedMemory1>>>(dIn, dOut, dBlockSums, N);
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("KERNEL 1: result = %.3f ms\n", ms);
    }

    if (blocks > 1) {

        // KERNEL 2
        int threads2 = 1;
        
        while (2 * threads2 < blocks) {
            threads2 <<= 1;
        }
        int M2 = 2 * threads2;
        int blocks2 = 1;
        size_t sharedMemory2 = (M2 + M2/NUM_BANKS) * sizeof(int);

        CUDA_OK(cudaEventRecord(start));
        exclusiveBlockBlelloch<<<blocks2, threads2, sharedMemory2>>>(dBlockSums, dBlockOffsets, /*blockSums=*/nullptr, /*N=*/blocks);
        CUDA_OK(cudaEventRecord(stop));
        CUDA_OK(cudaEventSynchronize(stop));
        {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            printf("KERNEL 2: result = %.3f ms\n", ms);
        }
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());

        // KERNEL 3
        CUDA_OK(cudaEventRecord(start));
        addBlockOffsets<<<blocks, threads>>>(dOut, dBlockOffsets, N);
        CUDA_OK(cudaEventRecord(stop));
        CUDA_OK(cudaEventSynchronize(stop));
        {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            printf("KERNEL 3: result = %.3f ms\n", ms);
        }
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }

    // copying D->H and compare
    CUDA_OK(cudaMemcpy(hOut.data(), dOut, N*sizeof(int), cudaMemcpyDeviceToHost));
    exclusiveScanCPU(hIn.data(), hRef.data(), N);

    printf("Vectors: \n");
    printf("GPU[0..15]: "); for (int i=0;i<16;i++) printf("%d ", hOut[i]); puts("");
    printf("CPU[0..15]: "); for (int i=0;i<16;i++) printf("%d ", hRef[i]); puts("");
    printf("GPU[123456]=%d, CPU[123456]=%d\n", hOut[123456], hRef[123456]);

    bool ok = true;
    for (int i = 0; i < N; ++i) if (hOut[i] != hRef[i]) ok = false;
    printf("check: %s\n", ok ? "OK" : "MISMATCH");

    cudaFree(dIn); cudaFree(dOut);
    cudaFree(dBlockSums); cudaFree(dBlockOffsets);
    return 0;
}