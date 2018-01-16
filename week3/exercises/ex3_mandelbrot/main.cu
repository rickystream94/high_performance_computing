#include <stdio.h>
#include <stdlib.h>
#include "mandelgpu.h"
#include "writepng.h"
#include <omp.h>
#include <helper_cuda.h>
#define NUM_THREADS 1
#define NUM_BLOCKS 1

int main(int argc, char *argv[])
{

    int width, height;
    int max_iter;
    int *h_image, *d_image;
    double ts, te;

    width = 4096;
    height = 4096;
    max_iter = 400;

    // command line argument sets the dimensions of the image
    if (argc == 2)
        width = height = atoi(argv[1]);

    // Allocate memory on host and device
    cudaMallocHost((void **)&h_image, width * height * sizeof(int));
    cudaMalloc((void **)&d_image, width * height * sizeof(int));

    if (d_image == NULL || h_image == NULL)
    {
        fprintf(stderr, "memory allocation failed!\n");
        return (1);
    }

    // Start timer
    ts = omp_get_wtime();

    // Launch kernel (only using 1 thread ATM)
    dim3 threadsPerBlock(64,64); // 64*64 = 4096 threads in total (1 per pixel)
    mandel<<<NUM_BLOCKS, threadsPerBlock>>>(width, height, d_image, max_iter);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    cudaMemcpy(h_image, d_image, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    // End timer
    te = omp_get_wtime() - ts;

    // Save pic
    writepng("mandelbrot.png", h_image, width, height);

    printf("%f\n", te);

    // Cleanup
    cudaFreeHost(h_image);
    cudaFree(d_image);

    return (0);
}
