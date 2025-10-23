// project 11 - Tiled GEMM - main.cppd
#include <cstdio> 
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>

// All the same as in 10 - Naive GEMM - main.cpp
void gemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) { 
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int M = 512;
    const int N = 512;
    const int K = 512;

    std::vector<float> A(M * K), B(K * N), C(M * N);
    for (auto& x : A) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& x : B) x = static_cast<float>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();

    gemmCPU(A.data(), B.data(), C.data(), M, N, K);

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double flops = 2.0 * M * N * K;
    double gpu_gflops = flops / (ms / 1000.0) / 1e9;
    printf("CPU: %.3f ms, %.1f GFLOP/s\n", ms, gpu_gflops);
}