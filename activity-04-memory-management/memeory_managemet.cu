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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {

    const int N = 1000;

    // =================================
    // 1. HOST MEMORY ALLOCATION
    // =================================

    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];

    // Initialize CPU arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    cout << "Host memory allocated." << endl;

    // =================================
    // 2. DEVICE MEMORY ALLOCATION
    // =================================

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_C, N * sizeof(int));

    cout << "Device memory allocated." << endl;

    // =================================
    // 3. HOST TO DEVICE TRANSFER
    // =================================

    cudaMemcpy(
        d_A,
        h_A,
        N * sizeof(int),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B,
        N * sizeof(int),
        cudaMemcpyHostToDevice
    );

    cout << "Data transferred: CPU -> GPU" << endl;

    // =================================
    // 4. GPU EXECUTION
    // =================================

    int threadsPerBlock = 256;

    int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaDeviceSynchronize();

    cout << "GPU computation completed." << endl;

    // =================================
    // 5. DEVICE TO HOST TRANSFER
    // =================================

    cudaMemcpy(
        h_C,
        d_C,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    cout << "Data transferred: GPU -> CPU" << endl;

    // =================================
    // 6. VALIDATION
    // =================================

    bool passed = true;

    for (int i = 0; i < N; i++) {

        if (h_C[i] != h_A[i] + h_B[i]) {
            passed = false;
            break;
        }
    }

    cout << "\nFirst 10 results:" << endl;

    for (int i = 0; i < 10; i++) {
        cout << h_A[i] << " + "
             << h_B[i] << " = "
             << h_C[i] << endl;
    }

    cout << "\nValidation: "
         << (passed ? "PASSED" : "FAILED")
         << endl;

    // =================================
    // 7. DEVICE MEMORY RELEASE
    // =================================

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cout << "\nDevice memory released." << endl;

    // =================================
    // 8. HOST MEMORY RELEASE
    // =================================

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cout << "Host memory released." << endl;

    return passed ? 0 : 1;
}