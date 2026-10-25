// we assign one gpu thread to each array element
# include <iostream>
# include <cuda_runtime.h>

const int N = 8;

// this function is executed on the GPU and adds two vectors
__global__ void vectorAdd(const int *a, const int *b, int *c, int n) {
    
    int i = threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


int main()
{
    int h_a[N] = {10, 20, 30, 40, 50, 60, 70, 80};
    int h_b[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_c[N];

    int *d_a, *d_b, *d_c;

    size_t bytes = sizeof(int) * N;

    // allocate memory on gpu
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy data from cpu to gpu
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    //launch 1 block containg 8 gpu threads
    vectorAdd<<<1, N>>>(d_a, d_b, d_c, N);

    // wait for gpu to finish before accessing on host
    cudaDeviceSynchronize();

    //copy results from gpu to cpu
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    // free gpu memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
