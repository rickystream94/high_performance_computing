extern "C" {
#include <cblas.h>
#include <math.h>
#include <stdio.h>
}
#include "matmultgpu.h"
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define BLOCK_SIZE 16
#define TILE_SIZE 4

extern "C" {
void matmult_nat(int m, int n, int k, double *A, double *B, double *C)
{
    // Initializing C
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            C[i * n + j] = 0.0;

    // Perform multiplication A*B
    int t, q;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            for (q = 0, t = 0; t < k && q < k; t++, q++)
                C[i * n + j] += A[i * k + q] * B[t * n + j];
}

void matmult_lib(int m, int n, int k, double *A, double *B, double *C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 0, C, n);
}

void matmult_gpu1(int m, int n, int k, double *h_A, double *h_B, double *h_C)
{
    double *d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));

    if (d_A == NULL || d_B == NULL || d_C == NULL)
    {
        fprintf(stderr, "memory allocation failed!\n");
        return;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    matmult_kernel_gpu1<<<1,1>>>(d_A,d_B,d_C,m,n,k);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matmult_gpu2(int m, int n, int k, double *h_A, double *h_B, double *h_C)
{
    double *d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));

    if (d_A == NULL || d_B == NULL || d_C == NULL)
    {
        fprintf(stderr, "memory allocation failed!\n");
        return;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE); // e.g. 16*16 = 256 threads in total
    dim3 num_blocks(ceil((double)n/threads_per_block.y),ceil((double)m/threads_per_block.x));
    matmult_kernel_gpu2<<<num_blocks,threads_per_block>>>(d_A,d_B,d_C,m,n,k);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matmult_gpu3(int m, int n, int k, double *h_A, double *h_B, double *h_C)
{
    double *d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));

    if (d_A == NULL || d_B == NULL || d_C == NULL)
    {
        fprintf(stderr, "memory allocation failed!\n");
        return;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE); // e.g. 16*16 = 256 threads in total
    //dim3 num_blocks(ceil((double)n/(threads_per_block.x*2)),ceil((double)m/(threads_per_block.y))); // RIGHT NEIGHBOR VERSION
    dim3 num_blocks(ceil((double)n/(threads_per_block.x)),ceil((double)m/(threads_per_block.y*2))); // BOTTOM NEIGHBOR VERSION
    matmult_kernel_gpu3<<<num_blocks,threads_per_block>>>(d_A,d_B,d_C,m,n,k);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matmult_gpu4(int m, int n, int k, double *h_A, double *h_B, double *h_C)
{
    double *d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));

    if (d_A == NULL || d_B == NULL || d_C == NULL)
    {
        fprintf(stderr, "memory allocation failed!\n");
        return;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE); // e.g. 16*16 = 256 threads in total
    dim3 num_blocks(ceil((double)n/(threads_per_block.x)),ceil((double)m/(threads_per_block.y*TILE_SIZE)));
    matmult_kernel_gpu4<<<num_blocks,threads_per_block>>>(d_A,d_B,d_C,m,n,k);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matmult_gpu5(int m, int n, int k, double *h_A, double *h_B, double *h_C)
{
    double *d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));

    if (d_A == NULL || d_B == NULL || d_C == NULL)
    {
        fprintf(stderr, "memory allocation failed!\n");
        return;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE); // e.g. 16*16 = 256 threads in total
    dim3 num_blocks(n/threads_per_block.x,m/threads_per_block.y);
    matmult_kernel_gpu5<<<num_blocks,threads_per_block>>>(d_A,d_B,d_C,m,n,k);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matmult_gpulib(int m, int n, int k, double *h_A, double *h_B, double *h_C)
{
    const double alpha = 1.0, beta = 0.0;
    double *d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));

    if (d_A == NULL || d_B == NULL || d_C == NULL)
    {
        fprintf(stderr, "memory allocation failed!\n");
        return;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);


    // Create handle for CUBLAS
    cublasHandle_t handle;
    if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS)
    {
        printf("Error initializing CUDA runtime.\n");
        return;
    }
    
    // Kernel invocation
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

    // Destroy handle
    cublasDestroy(handle);

    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
}