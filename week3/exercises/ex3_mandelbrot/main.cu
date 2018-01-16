#include <stdio.h>
#include <stdlib.h>
#include "mandelgpu.h"
#include "writepng.h"
#include <omp.h>
#include <helper_cuda.h>
#define THREADS_PER_BLOCK 16

int main(int argc, char *argv[])
{

    int width, height, max_iter, threads_per_block;
    int *h_image, *d_image;
    double ts_warmup, te_warmup, ts_kernel, te_kernel, ts_copy, te_copy, te_total;

    width = 4096;
    height = 4096;
    max_iter = 400;

    // command line argument sets the dimensions of the image
    if (argc == 2)
        width = height = atoi(argv[1]);
        //threads_per_block = atoi(argv[1]);

    // Allocate memory on host and device
    ts_warmup = omp_get_wtime();
    cudaMallocHost((void **)&h_image, width * height * sizeof(int));
    cudaMalloc((void **)&d_image, width * height * sizeof(int));
    te_warmup = omp_get_wtime() - ts_warmup;

    if (d_image == NULL || h_image == NULL)
    {
        fprintf(stderr, "memory allocation failed!\n");
        return (1);
    }

    // Start timer
    ts_kernel = omp_get_wtime();

    // Launch kernel (only using 1 thread ATM)
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK); // e.g. 16*16 = 256 threads in total
    dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);
    mandel<<<numBlocks, threadsPerBlock>>>(width, height, d_image, max_iter);
    checkCudaErrors(cudaDeviceSynchronize());

    // End timer
    te_kernel = omp_get_wtime() - ts_kernel;

    // Copy result back to host
    ts_copy = omp_get_wtime();
    cudaMemcpy(h_image, d_image, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    te_copy = omp_get_wtime() - ts_copy;

    // Count total time
    te_total = omp_get_wtime() - ts_warmup;

    // Save pic
    writepng("mandelbrot.png", h_image, width, height);

    printf("--- Threads per block: %d ---\n--- Image size: %d ---\nTotal warmup time: %f\nTotal Kernel time: %f\nTotal MemCopy time: %f\nTotal Time: %f\n\n", THREADS_PER_BLOCK,width,te_warmup,te_kernel,te_copy,te_total);

    // Cleanup
    cudaFreeHost(h_image);
    cudaFree(d_image);

    return (0);
}
