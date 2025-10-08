// project 02 - Reduction - main.cpp
#include <cstdio>
#include <vector> 
#include <chrono>
using namespace std::chrono;

long long reduceCPU(const int* A, int N) {
    long long s = 0;
    for (int i = 0; i < N; i++)
    {
        s += A[i];
    }
    return s;
}

int main() {
    int N = 1 << 20; // estimated 1 million
    std::vector<int> A(N, 1);

    auto start = high_resolution_clock::now();

    long long ref = reduceCPU(A.data(), N);

    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start).count();

    printf("CPU sum = %lld, time = %lld us\n", ref, (long long)us);

}