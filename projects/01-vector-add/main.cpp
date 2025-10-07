// project 01 - main.cpp
#include <cstdio>
#include <vector>
#include <cassert>
#include <chrono>
using namespace std::chrono;

bool vectorAddCPU(const int* A, const int* B, int* C, int N) {
    for (int i = 0; i < N; i++){
        C[i] = A[i] + B[i];
    }

    return true;
}

int main() {
    constexpr int N = 1 << 28;
    std::vector<int> A(N), B(N), C(N), Ref(N);

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    auto start = high_resolution_clock::now();

    auto result = vectorAddCPU(A.data(), B.data(), Ref.data(), N);
    printf("Result: %d", (int)result);
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start).count();

    printf("CPU Time: %lld ms\n", duration);

    for (int i = 0; i < 5; i++) {
        printf("%d ", Ref[i]);
    }
    printf("\n");

    return 0;
}