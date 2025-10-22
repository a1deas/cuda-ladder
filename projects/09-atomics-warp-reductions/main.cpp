// project 09 - Atomics & Warp Reductions - main.cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <cstdio>

int main() { 
    const int N = 100000000;
    std::vector<float> data(N, 1.0f);

    auto start = std::chrono::high_resolution_clock::now();

    float sum = std::accumulate(data.begin(), data.end(), 0.0f);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms = end - start;
    std::printf("CPU result:\n Sum: %.1f\n Time: %.3f ms\n", sum, ms.count());
}