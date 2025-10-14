// project 04 - Microbench Coalescing - main.cu
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

// defined constants
#define N (1 << 24)
#define THREADS 256
#define BLOCKS ((N + THREADS - 1) / THREADS)

// stride = 1, ideal access: data[idx], all threads go one by one
__global__ void coalesced(float* data, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int gsize = gridDim.x * blockDim.x;
    for (int i = gtid; i < n; i += gsize) {
        float v = data[i];
        data[i] = v * 2.0f;
    }
}

// every thread reads with a stride step
// every thread goes into another 128-byte segment
__global__ void nonCoalesced(float* data, int n, int stride) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int gsize = gridDim.x * blockDim.x;

    for (int i = gtid; i < n; i += gsize) {
        int access = (i * stride) & (N - 1);
        float v = data[access];
        data[access] = v * 2.0f;
    }

}

// helper function for kernel launch
template <typename Kernel, typename... Args>
float runKernel(const char* name, dim3 blocks, dim3 threads, Kernel kernel, Args... args) {
    // create events and syncronize devices
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));

    // start recording
    CUDA_OK(cudaEventRecord(start));

    // launch kernel
    kernel<<<blocks, threads>>>(args...);
    CUDA_OK(cudaGetLastError());

    // stop recording
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
        
    // get result
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double bytes = static_cast<double>(N) * sizeof(float) * 2;
    double gbps  = bytes / (ms / 1000.0) / 1e9;                
    std::printf("%-22s | %8.3f ms | %7.2f GB/s\n", name, ms, gbps);

    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    return ms;
}

int main() {
    // create device buffer and provide memory
    float* dIn;
    size_t size = N * sizeof(float);
    CUDA_OK(cudaMalloc(&dIn, size));

    // fill host buffer with i
    std::vector<float> hIn(N);
    for (int i = 0; i < N; i++) { 
        hIn[i] = i;
    }

    // warm-up
    coalesced<<<BLOCKS, THREADS>>>(dIn, N);
    cudaDeviceSynchronize();
    coalesced<<<BLOCKS, THREADS>>>(dIn, N);
    cudaDeviceSynchronize();

    // reset helper, so every launch will from zero
    auto resetDevice = [&](float value) {
        std::fill(hIn.begin(), hIn.end(), value);
        // copying from host to device
        CUDA_OK(cudaMemcpy(dIn, hIn.data(), size, cudaMemcpyHostToDevice));
    };
    // reset device
    resetDevice(1.0f);

    // run kernel and synchronize
    runKernel("coalesced", dim3(BLOCKS), dim3(THREADS), coalesced, dIn, N); 
    cudaDeviceSynchronize();

    // run non-coalesced version
    for (int stride = 1; stride <= 32; stride *= 2) {
        resetDevice(1.0f);
        char label[64];
        std::snprintf(label, sizeof(label), "nonCoalesced s=%-2d", stride);
        runKernel(label, dim3(BLOCKS), dim3(THREADS), nonCoalesced, dIn, N, stride);
    }

    // copying back from device to host and freeing memory
    CUDA_OK(cudaMemcpy(hIn.data(), dIn, size, cudaMemcpyDeviceToHost));
    cudaFree(dIn);
}