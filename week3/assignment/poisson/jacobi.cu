#include <stdlib.h>
#include <omp.h>
#include <helper_cuda.h>
#include "matrixlib.h"
#include <math.h>
#include "jacobigpu.h"
#define DOMAIN_LENGTH 2
#define BLOCK_SIZE 16

void jacobi_1(int N, int k_max)
{
    // Variables declaration
    int k;
    double *h_U, *h_f, *d_U, *d_U_old, *d_U_new, *d_f, *temp_ptr;
    double delta, ts, te;

    // 2. allocate memory for the necessary data fields
    cudaMalloc((void **)&d_U, N * N * sizeof(double));
    cudaMalloc((void **)&d_U_new, N * N * sizeof(double));
    cudaMalloc((void **)&d_f, N * N * sizeof(double));
    cudaMallocHost((void **)&h_U, N * N * sizeof(double));
    cudaMallocHost((void **)&h_f, N * N * sizeof(double));

    // Calculate delta
    delta = (double)(DOMAIN_LENGTH) / (N - 1);

    // 3. initialize the fields with your start and boundary conditions
    init_f_matrix(N, h_f, delta);
    init_U_matrix(N, h_U);

    // 4. copy data from host to device
    cudaMemcpy(d_U,h_U,N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f,h_f,N * N * sizeof(double), cudaMemcpyHostToDevice);

    // 5. call kernel iterator
    // Get starting time
    ts = omp_get_wtime();

    // Assign mat_old with the initial guess (k = 0 iteration)
    d_U_old = d_U;
    for(k = 0; k < k_max; k++)
    {
        jacobi_kernel1<<<1,1>>>(N, d_U_old, d_U_new, d_f, delta);
        checkCudaErrors(cudaDeviceSynchronize());

        // Swap the pointers on the CPU
        {
            temp_ptr = d_U_old;
            d_U_old = d_U_new;
            d_U_new = temp_ptr;
        }
    }
    
    // Get ending time
    te = omp_get_wtime() - ts;

    // Copy result back to host (notice that d_U_old will have the last good result!)
    cudaMemcpy(h_U, d_U_old, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // 6. print results, e.g. timings, data, etc
    //print_matrix(N, h_U); // Only for debugging
    printf("%f\n", te);

    // 7. Cleanup
    cudaFreeHost(h_U);
    cudaFreeHost(h_f);
    cudaFree(d_f);
    cudaFree(d_U);
    cudaFree(d_U_new);
}

void jacobi_2(int N, int k_max)
{
    // Variables declaration
    int k;
    double *h_U, *h_f, *d_U, *d_U_old, *d_U_new, *d_f, *temp_ptr;
    double delta, ts, te;

    // 2. allocate memory for the necessary data fields
    cudaMalloc((void **)&d_U, N * N * sizeof(double));
    cudaMalloc((void **)&d_U_new, N * N * sizeof(double));
    cudaMalloc((void **)&d_f, N * N * sizeof(double));
    cudaMallocHost((void **)&h_U, N * N * sizeof(double));
    cudaMallocHost((void **)&h_f, N * N * sizeof(double));

    // Calculate delta
    delta = (double)(DOMAIN_LENGTH) / (N - 1);

    // 3. initialize the fields with your start and boundary conditions
    init_f_matrix(N, h_f, delta);
    init_U_matrix(N, h_U);

    // 4. copy data from host to device
    cudaMemcpy(d_U,h_U,N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f,h_f,N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Define GPU thread blocks dimensions
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE); // e.g. 16*16 = 256 threads in total
    dim3 num_blocks(ceil((double)N/threads_per_block.x), ceil((double)N/threads_per_block.y));

    // 5. call kernel iterator
    // Get starting time
    ts = omp_get_wtime();

    // Assign mat_old with the initial guess (k = 0 iteration)
    d_U_old = d_U;
    for(k = 0; k < k_max; k++)
    {
        jacobi_kernel2<<<num_blocks,threads_per_block>>>(N, d_U_old, d_U_new, d_f, delta);
        checkCudaErrors(cudaDeviceSynchronize());

        // Swap the pointers on the CPU
        {
            temp_ptr = d_U_old;
            d_U_old = d_U_new;
            d_U_new = temp_ptr;
        }
    }
    
    // Get ending time
    te = omp_get_wtime() - ts;

    // Copy result back to host (notice that d_U_old will have the last good result!)
    cudaMemcpy(h_U, d_U_old, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // 6. print results, e.g. timings, data, etc
    //print_matrix(N, h_U); // Only for debugging
    printf("%f\n", te);

    // 7. Cleanup
    cudaFreeHost(h_U);
    cudaFreeHost(h_f);
    cudaFree(d_f);
    cudaFree(d_U);
    cudaFree(d_U_new);
}

void jacobi_3(int N, int k_max)
{
    // Variables declaration
    int k;
    double *h_U, *h_f, *d0_U, *d1_U, *d0_U_old, *d1_U_old, *d0_U_new, *d1_U_new, *d0_f, *d1_f, *temp_ptr;
    double delta, ts, te;

    // Allocate host memory
    cudaMallocHost((void **)&h_U, N * N * sizeof(double));
    cudaMallocHost((void **)&h_f, N * N * sizeof(double));

    // Calculate delta
    delta = (double)(DOMAIN_LENGTH) / (N - 1);

    // 3. initialize the fields with your start and boundary conditions
    init_f_matrix(N, h_f, delta);
    init_U_matrix(N, h_U);

    // Define GPU thread blocks dimensions
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE); // e.g. 16*16 = 256 threads in total
    dim3 num_blocks(ceil((double)N/(threads_per_block.x*2)), ceil((double)N/(threads_per_block.y*2)));

    // DEVICE 0
    cudaSetDevice(0);
    // 2. allocate memory for the necessary data fields
    cudaMalloc((void **)&d0_U, N/2 * N * sizeof(double));
    cudaMalloc((void **)&d0_U_new, N/2 * N * sizeof(double));
    cudaMalloc((void **)&d0_f, N/2 * N * sizeof(double));

    // 4. copy data from host to device
    cudaMemcpy(d0_U,h_U,N/2 * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d0_f,h_f,N/2 * N * sizeof(double), cudaMemcpyHostToDevice);

    // DEVICE 1
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);
    // 2. allocate memory for the necessary data fields
    cudaMalloc((void **)&d1_U, N/2 * N * sizeof(double));
    cudaMalloc((void **)&d1_U_new, N/2 * N * sizeof(double));
    cudaMalloc((void **)&d1_f, N/2 * N * sizeof(double));

    // 4. copy data from host to device
    cudaMemcpy(d1_U,h_U + N/2,N/2 * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d1_f,h_f + N/2,N/2 * N * sizeof(double), cudaMemcpyHostToDevice);

    // Get starting time
    ts = omp_get_wtime();

    // Assign mat_old with the initial guess (k = 0 iteration)
    d0_U_old = d0_U;
    d1_U_old = d1_U;
    for(k = 0; k < k_max; k++)
    {
        cudaSetDevice(0);
        jacobi_kernel_multigpu_0<<<num_blocks,threads_per_block>>>(N, d0_U_old, d0_U_new, d0_f, delta);
        cudaSetDevice(1);
        jacobi_kernel_multigpu_1<<<num_blocks,threads_per_block>>>(N, d1_U_old, d1_U_new, d1_f, delta);
        checkCudaErrors(cudaDeviceSynchronize());

        // Swap the pointers on the CPU
        temp_ptr = d0_U_old;
        d0_U_old = d0_U_new;
        d0_U_new = temp_ptr;

        temp_ptr = d1_U_old;
        d1_U_old = d1_U_new;
        d1_U_new = temp_ptr;
    }
    
    // Get ending time
    te = omp_get_wtime() - ts;

    // Copy result back to host (notice that d_U_old will have the last good result!)
    cudaMemcpy(h_U, d0_U_old, N/2 * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U + N/2, d1_U_old, N/2 * N * sizeof(double), cudaMemcpyDeviceToHost);

    // 6. print results, e.g. timings, data, etc
    //print_matrix(N, h_U); // Only for debugging
    printf("%f\n", te);

    // 7. Cleanup
    cudaFreeHost(h_U);
    cudaFreeHost(h_f);
    cudaSetDevice(0);
    cudaFree(d0_f);
    cudaFree(d0_U);
    cudaFree(d0_U_new);
    cudaSetDevice(1);
    cudaFree(d1_f);
    cudaFree(d1_U);
    cudaFree(d1_U_new);
}