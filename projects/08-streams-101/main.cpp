// project 08 - CUDA Streams(Streams 101) - main.cpp
#include <vector>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 50000000;
    int iters = (argc > 2) ? std::atoi(argv[2]) : 100;
    const float a = 3.0f;

    std::vector<float> x(N, 1.0f), y(N, 2.0f);

    auto start = std::chrono::high_resolution_clock::now();
    
    // light version
    // for (int i = 0; i < N; ++i) y[i] = a * x[i] + y[i];
    
    // heavy version
    for (int i = 0; i < N; ++i) {
        float acc = 0.0f;
        for (int k = 0; k < iters; ++k) {
            acc = std::fma(a, x[i], acc);
        }
        y[i] = acc + y[i];
    }

    auto stop = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    printf("CPU time: %f ms\n", ms);

    bool ok = true;
    float expected = 2.0f + iters * a * 1.0f;
    for (int i : {0, N/2, N-1}) {
        if (std::fabs(y[i] - expected) > 1e-4f) { ok = false; break; }
    }
    printf("Validate: %s", (ok ? "OK" : "FAIL")); 
    return ok ? 0 : 1;
}