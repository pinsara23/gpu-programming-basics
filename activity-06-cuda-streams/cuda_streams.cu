#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

// CUDA error checking
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error = call;                             \
        if (error != cudaSuccess) {                           \
            cerr << "CUDA Error: "                            \
                 << cudaGetErrorString(error) << endl;        \
            return 1;                                         \
        }                                                     \
    } while (0)

// CUDA Kernel
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

    const int N = 1 << 20;       // 1,048,576 elements
    const int STREAM_COUNT = 16;

    const size_t totalBytes = N * sizeof(int);

    // ==========================================
    // 1. ALLOCATE PINNED HOST MEMORY
    // ==========================================

    int *h_A, *h_B, *h_C;

    CUDA_CHECK(cudaMallocHost(&h_A, totalBytes));
    CUDA_CHECK(cudaMallocHost(&h_B, totalBytes));
    CUDA_CHECK(cudaMallocHost(&h_C, totalBytes));

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    cout << "Pinned host memory allocated." << endl;

    // ==========================================
    // 2. CREATE CUDA STREAMS
    // ==========================================

    cudaStream_t streams[STREAM_COUNT];

    for (int i = 0; i < STREAM_COUNT; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    cout << STREAM_COUNT << " CUDA streams created." << endl;

    // ==========================================
    // 3. ALLOCATE DEVICE BUFFERS
    // ==========================================

    const int chunkSize = (N + STREAM_COUNT - 1)
                          / STREAM_COUNT;

    const size_t chunkBytes = chunkSize * sizeof(int);

    int *d_A[STREAM_COUNT];
    int *d_B[STREAM_COUNT];
    int *d_C[STREAM_COUNT];

    for (int i = 0; i < STREAM_COUNT; i++) {

        CUDA_CHECK(cudaMalloc(&d_A[i], chunkBytes));
        CUDA_CHECK(cudaMalloc(&d_B[i], chunkBytes));
        CUDA_CHECK(cudaMalloc(&d_C[i], chunkBytes));
    }

    // ==========================================
    // 4. START PERFORMANCE MEASUREMENT
    // ==========================================

    auto start = high_resolution_clock::now();

    // ==========================================
    // 5. SUBMIT WORK TO EACH STREAM
    // ==========================================

    const int threadsPerBlock = 256;

    for (int i = 0; i < STREAM_COUNT; i++) {

        int offset = i * chunkSize;

        int currentSize = min(chunkSize, N - offset);

        size_t currentBytes = currentSize * sizeof(int);

        int blocksPerGrid =
            (currentSize + threadsPerBlock - 1)
            / threadsPerBlock;

        // CPU -> GPU transfer
        CUDA_CHECK(cudaMemcpyAsync(
            d_A[i],
            h_A + offset,
            currentBytes,
            cudaMemcpyHostToDevice,
            streams[i]
        ));

        CUDA_CHECK(cudaMemcpyAsync(
            d_B[i],
            h_B + offset,
            currentBytes,
            cudaMemcpyHostToDevice,
            streams[i]
        ));

        // GPU computation
        vectorAddition<<<
            blocksPerGrid,
            threadsPerBlock,
            0,
            streams[i]
        >>>(
            d_A[i],
            d_B[i],
            d_C[i],
            currentSize
        );

        CUDA_CHECK(cudaGetLastError());

        // GPU -> CPU transfer
        CUDA_CHECK(cudaMemcpyAsync(
            h_C + offset,
            d_C[i],
            currentBytes,
            cudaMemcpyDeviceToHost,
            streams[i]
        ));
    }

    // ==========================================
    // 6. WAIT FOR ALL STREAMS
    // ==========================================

    for (int i = 0; i < STREAM_COUNT; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    auto stop = high_resolution_clock::now();

    double totalTime =
        duration<double, milli>(stop - start).count();

    // ==========================================
    // 7. VALIDATION
    // ==========================================

    bool passed = true;

    for (int i = 0; i < N; i++) {

        if (h_C[i] != h_A[i] + h_B[i]) {
            passed = false;
            break;
        }
    }

    // ==========================================
    // 8. DISPLAY RESULTS
    // ==========================================

    cout << "\n========== CUDA STREAM RESULTS ==========\n";

    cout << "Number of elements: " << N << endl;
    cout << "Number of streams: " << STREAM_COUNT << endl;
    cout << "Elements per stream (maximum): "
         << chunkSize << endl;

    cout << "Total GPU processing time: "
         << totalTime << " ms" << endl;

    cout << "Throughput: "
         << (N / (totalTime / 1000.0))
         << " elements/second" << endl;

    cout << "\nValidation: "
         << (passed ? "PASSED" : "FAILED")
         << endl;

    // ==========================================
    // 9. CLEANUP
    // ==========================================

    for (int i = 0; i < STREAM_COUNT; i++) {

        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));

        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    cout << "\nAll memory and streams released." << endl;

    return passed ? 0 : 1;
}