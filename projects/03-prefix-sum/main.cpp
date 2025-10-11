// project 03 - Prefix-sum/Scan - main.cpp
#include <vector> 
#include <cstdio> 
#include <chrono> 
using namespace std::chrono;

// Exclusive variant
void exclusiveScanCPU(const int* in, int* out, int N) {
    int local = 0;
    for (int i = 0; i < N; ++i) {
        out[i] = local;
        local += in[i];
    }
}

int main() { 
    int N = 1 << 20; // ~1M elements
    std::vector<int> x(N), y(N); // init two vectors
    for (int i = 0; i < N; ++i) {
        x[i] = i;
    }

    auto start = high_resolution_clock::now();

    exclusiveScanCPU(x.data(), y.data(), N);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start).count();

    printf("CPU Time = %lld ms\n", duration);

}