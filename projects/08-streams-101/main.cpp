// project 07 - CUDA Streams(Streams 101) - main.cpp
#include <vector>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 50000000;
    const float a = 3.0f;

    std::vector<float> x(N, 1.0f), y(N, 2.0f);

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < N; ++i) y[i] = a * x[i] + y[i];
    
    auto stop = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "CPU time: " << ms << " ms\\n";

    bool ok = true;
    for (int i : {0, N/2, N-1}) {
        float expected = 5.0f;
        if (std::fabs(y[i] - expected) > 1e-5f) { ok = false; break; }
    }
    std::cout << "Validate: " << (ok ? "OK" : "FAIL") << "\\n";
    return ok ? 0 : 1;
}