// project 05 - Shared Memory Tiles - main.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <vector> 

// macro to check for the CUDA errors
#define CUDA_OK(stmt) do {                                  \
    cudaError_t err = (stmt);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s at %s:%d\n",         \
            cudaGetErrorString(err), __FILE__, __LINE__);   \
        std::exit(1);                                       \
    }                                                       \
} while(0)

static constexpr int TILE_SIZE = 32;
static constexpr int N = 4096;
static constexpr int THREADS = TILE_SIZE;

// Naive, without shared memory
// It reads rows, but column writing -- not coalescing.
__global__ void transposeNaive(float* out, const float* in, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        out[x * n + y] = in[y * n + x];
    }
    //cudaGetLastError();
}

// Tiled realisation
__global__ void transposeTiled(float* out, const float* in, int n) { 
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to prevent bank conflict

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // coalescing reading 
    if(x < n && y < n) { 
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    }

    // so all threads will wait for each other.
    __syncthreads();

    // transpose to the global memory(coaslesced too)
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < n && y < n) {
        out[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}

template<typename Kernel>
float runKernel(const char* name, Kernel kernel, float* out, float* in, int n) {
    // Init events
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    
    // Init threads and blocks
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks( (n + TILE_SIZE - 1) / TILE_SIZE,
        (n + TILE_SIZE - 1) / TILE_SIZE );

    // Warm-up
    transposeTiled<<<blocks, threads>>>(out, in, n);
    cudaDeviceSynchronize();

    // Start recording
    cudaEventRecord(start);
    
    // Launch kernel
    kernel<<<blocks, threads>>>(out, in, n);
    CUDA_OK(cudaGetLastError());

    // Stop recording and get result
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    double bytes = double(n) * n * sizeof(float) * 2;
    double gbps = bytes / (ms / 1000.0) / 1e9;
    printf("%-18s | %7.3f ms | %7.2f GB/s\n", name, ms, gbps);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    size_t bytes = N * N * sizeof(float);
    // Init host buffers and fill hIn
    float* hIn = new float[N * N];
    float* hOut = new float[N * N];
    for (int i = 0; i < N * N; ++i) {
        hIn[i] = float(i);
    }

    // Init device buffers and provide memory
    float* dIn;
    float* dOut;
    CUDA_OK(cudaMalloc(&dIn, bytes));
    CUDA_OK(cudaMalloc(&dOut, bytes));
    
    // Copying memory from Host to 
    CUDA_OK(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));
    
    runKernel("Naive", transposeNaive, dOut, dIn, N);
    runKernel("Tiled", transposeTiled, dOut, dIn, N);

    CUDA_OK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dIn), cudaFree(dOut);
    delete[] hIn;
    delete[] hOut;
}