#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

// Check CUDA errors
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error = call;                             \
        if (error != cudaSuccess) {                           \
            cerr << "CUDA Error: "                            \
                 << cudaGetErrorString(error) << endl;        \
            return 1;                                         \
        }                                                     \
    } while (0)

// GPU Kernel
__global__ void vectorAddition(
    const int *A,
    const int *B,
    int *C,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {

    const int N = 1000000;  // 1,048,576 elements

    const size_t bytes = N * sizeof(int);

    // =====================================
    // 1. HOST MEMORY ALLOCATION
    // =====================================

    vector<int> A(N);
    vector<int> B(N);
    vector<int> C_CPU(N);
    vector<int> C_GPU(N);

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    // =====================================
    // 2. CPU PERFORMANCE MEASUREMENT
    // =====================================

    auto cpuStart = high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        C_CPU[i] = A[i] + B[i];
    }

    auto cpuStop = high_resolution_clock::now();

    double cpuTime =
        duration<double, milli>(cpuStop - cpuStart).count();

    // =====================================
    // 3. GPU MEMORY ALLOCATION
    // =====================================

    int *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Create CUDA events
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // =====================================
    // 4. CPU TO GPU TRANSFER TIMING
    // =====================================

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(
        d_A, A.data(), bytes, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, B.data(), bytes, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2dTime = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &h2dTime, start, stop
    ));

    // =====================================
    // 5. GPU KERNEL CONFIGURATION
    // =====================================

    int threadsPerBlock = 256;

    int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    // Warm-up execution
    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // =====================================
    // 6. GPU KERNEL TIMING
    // =====================================

    CUDA_CHECK(cudaEventRecord(start));

    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, N
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelTime = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &kernelTime, start, stop
    ));

    // =====================================
    // 7. GPU TO CPU TRANSFER TIMING
    // =====================================

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(
        C_GPU.data(), d_C, bytes, cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2hTime = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &d2hTime, start, stop
    ));

    // =====================================
    // 8. PERFORMANCE CALCULATIONS
    // =====================================

    double gpuTotalTime =
        h2dTime + kernelTime + d2hTime;

    double kernelSpeedup =
        cpuTime / kernelTime;

    double endToEndSpeedup =
        cpuTime / gpuTotalTime;

    // =====================================
    // 9. VALIDATION
    // =====================================

    bool passed = true;

    for (int i = 0; i < N; i++) {

        if (C_CPU[i] != C_GPU[i]) {
            passed = false;
            break;
        }
    }

    // =====================================
    // 10. DISPLAY RESULTS
    // =====================================

    cout << "\n========== PERFORMANCE RESULTS ==========\n";

    cout << "Number of elements: " << N << endl;

    cout << "\nCPU computation time: "
         << cpuTime << " ms" << endl;

    cout << "\nGPU H2D transfer time: "
         << h2dTime << " ms" << endl;

    cout << "GPU kernel execution time: "
         << kernelTime << " ms" << endl;

    cout << "GPU D2H transfer time: "
         << d2hTime << " ms" << endl;

    cout << "\nGPU total time: "
         << gpuTotalTime << " ms" << endl;

    cout << "\nKernel-only speedup: "
         << kernelSpeedup << "x" << endl;

    cout << "End-to-end speedup: "
         << endToEndSpeedup << "x" << endl;

    cout << "\nValidation: "
         << (passed ? "PASSED" : "FAILED") << endl;

    // =====================================
    // 11. CLEANUP
    // =====================================

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? 0 : 1;
}