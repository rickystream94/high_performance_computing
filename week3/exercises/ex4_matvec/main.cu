#include <stdio.h>
#include <stdlib.h>
#include "matvec.h"
#include "matrixlib.h"
#include <helper_cuda.h>
#define BLOCK_SIZE 16
#define NUM_BLOCKS 16

int main(int argc, char **argv)
{
    int m, n;
    double *h_A, *d_A, *h_v, *d_v, *h_w, *d_w;

    // Fetch command line parameters, otherwise use default
    if (argc == 3)
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }
    else
    {
        m = 5;
        n = 5;
    }

    // Allocate memory on host and device
    cudaMallocHost((void **)&h_A, m * n * sizeof(double)); // A is m x n
    cudaMallocHost((void **)&h_v, n * sizeof(double));     // v is n x 1
    cudaMallocHost((void **)&h_w, m * sizeof(double));     // w is m x 1
    cudaMalloc((void **)&d_A, m * n * sizeof(double));
    cudaMalloc((void **)&d_v, n * sizeof(double));
    cudaMalloc((void **)&d_w, m * sizeof(double));

    if (h_A == NULL || h_v == NULL || h_w == NULL || d_A == NULL || d_v == NULL || d_w == NULL)
    {
        fprintf(stderr, "memory allocation failed!\n");
        return (1);
    }

    // Init matrices on host
    init_matrix(m, n, h_A, 1.0);
    init_vector(n, h_v, 2.0);

    // Print input
    print_matrix(m,n,h_A);
    print_vector(n,h_v);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, n * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel (only using 1 thread ATM)
    //dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE); // For 2D problems, e.g. 16*16 = 256 threads per block
    //dim3 numBlocks(m / threadsPerBlock.x, n / threadsPerBlock.y);
    //mat_x_vec_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_v, d_w, m, n);
    // The problem is 1D!!!
    mat_x_vec_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_A, d_v, d_w, m, n); // For 1D problems, e.g. 16*16 = 256 threads in total
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    cudaMemcpy(h_w, d_w, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Print result
    print_vector(m,h_w);

    // Cleanup
    cudaFreeHost(h_A);
    cudaFreeHost(h_v);
    cudaFreeHost(h_w);
    cudaFree(d_A);
    cudaFree(d_v);
    cudaFree(d_w);

    return 0;
}