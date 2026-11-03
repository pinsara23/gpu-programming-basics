// CUDA organizes GPU execution into:
// - Grid – Collection of blocks.
// - Block – Collection of threads.
// - Thread – Individual execution unit.

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel
__global__ void vectorAddition(
    int *A,
    int *B,
    int *C,
    int n
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // current_block_index * threads_per_block + current_thread_index

    // Prevent accessing elements outside the array because we created 1024 threads for 1000 elements
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {

    const int N = 5000;

    int *A, *B, *C;

    int *d_A, *d_B, *d_C;

    // Allocate host memory
    A = new int[N];
    B = new int[N];
    C = new int[N];

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    // Allocate GPU memory
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_C, N * sizeof(int));

    // Copy input arrays to GPU
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Configure CUDA execution
    int threadsPerBlock = 256;

    int numberOfBlocks =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    cout << "Number of elements: " << N << endl;
    cout << "Threads per block: " << threadsPerBlock << endl;
    cout << "Number of blocks: " << numberOfBlocks << endl;

    // Launch CUDA kernel
    vectorAddition<<<numberOfBlocks, threadsPerBlock>>>(
        d_A, d_B, d_C, N
    );

    // Wait for GPU execution
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Validate results
    bool passed = true;

    for (int i = 0; i < N; i++) {

        if (C[i] != A[i] + B[i]) {
            passed = false;
            break;
        }
    }

    // Display first 10 results
    cout << "\nFirst 10 results:\n";

    for (int i = 0; i < 10; i++) {
        cout << A[i] << " + "
             << B[i] << " = "
             << C[i] << endl;
    }

    cout << "\nValidation: "
         << (passed ? "PASSED" : "FAILED")
         << endl;

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    delete[] A;
    delete[] B;
    delete[] C;

    return passed ? 0 : 1;
}