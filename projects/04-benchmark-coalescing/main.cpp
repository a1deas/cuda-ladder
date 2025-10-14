// project 04 - Microbench Coalescing - main.cpp
#include <cstdio> 
#include <vector>
#include <chrono>
using namespace std::chrono;

// stride = 1 -> coalesced
// stride = 32 -> non-coalesced
void cpuStrideAccess(std::vector<float>& data, int stride) {
    int N = data.size();
    for (int i = 0; i < N; i += stride) {
        data[i] *= 2.0f;
    }
}

int main() {
    const int N = 1 << 24; // ~16M elements
    std::vector<float> base(N, 1.0f); 

    // try different strides(1 -> 2 -> 4 -> 8 -> 16 -> 32)
    for (int stride = 1; stride <= 32; stride *= 2) {
        std::vector<float> data = base;

        auto start = high_resolution_clock::now();
        cpuStrideAccess(data, stride);
        auto stop = high_resolution_clock::now();

        double ms = duration_cast<microseconds>(stop - start).count() / 1000.0;
        size_t touched = (N + stride - 1) / stride;
        double bytes   = double(touched) * 2 * sizeof(float);
        double gbps    = bytes / (ms / 1000.0) / 1e9;
        printf("CPU stride(%2d) : %8.3f ms | %7.2f GB/s\n", stride, ms, gbps);

    }
}