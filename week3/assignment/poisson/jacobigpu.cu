#include <stdio.h>
#include "matrixlib.h"

__global__ void jacobi_kernel1(int N, double *mat_old, double *mat_new, double *f, double delta)
{
    int i, j;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            // Check boundary values (should be copied in the new matrix as they are)
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
                mat_new[i * N + j] = mat_old[i * N + j];
            else
            {
                // Calculate approximization
                mat_new[i * N + j] = (1.0 / 4.0) * (mat_old[i * N + (j - 1)] + mat_old[i * N + (j + 1)] + mat_old[(i + 1) * N + j] + mat_old[(i - 1) * N + j] + delta * delta * f[i * N + j]);
            }
        }
    } /* end of point approximization */
}

__global__ void jacobi_kernel2(int N, double *mat_old, double *mat_new, double *f, double delta)
{
    int i,j;

    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N || j >= N)
        return;

    // Check boundary values (should be copied in the new matrix as they are)
    if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
        mat_new[i * N + j] = mat_old[i * N + j];
    else
    {
        // Calculate approximization
        mat_new[i * N + j] = (1.0 / 4.0) * (mat_old[i * N + (j - 1)] + mat_old[i * N + (j + 1)] + mat_old[(i + 1) * N + j] + mat_old[(i - 1) * N + j] + delta * delta * f[i * N + j]);
    }
}

__global__ void jacobi_kernel_multigpu_0(int N, double *d_U_old, double *d_U_new, double *d_f, double delta)
{

}

__global__ void jacobi_kernel_multigpu_1(int N, double *d_U_old, double *d_U_new, double *d_f, double delta)
{
    
}