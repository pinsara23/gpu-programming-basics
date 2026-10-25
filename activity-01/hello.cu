#include <cstdio>
#include <cuda_runtime.h>

__global__ void helloGPU()
{
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main()
{
    // Launch one block containing eight GPU threads.
    helloGPU<<<1, 8>>>();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU execution to finish.
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("GPU execution error: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    printf("Hello from CPU!\n");

    return 0;
}