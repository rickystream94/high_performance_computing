extern "C" {
#include <stdio.h>
}
#define TILE_SIZE_2 2
__global__ void matmult_kernel_gpu1(double *A, double *B, double *C, int m, int n, int k)
{
    int i,j,e;

    // Initialize element
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            C[i*n+j] = 0.0;

    // Compute mult
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            for (e = 0; e < k; e++)
                C[i*n+j] += A[i*k+e] * B[e*n+j];
}

__global__ void matmult_kernel_gpu2(double *A, double *B, double *C, int m, int n, int k)
{
    int i,j,e;

    // Get (unique) thread index
    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i>=m || j>=n)
        return;

    // Initialize element
    double sum = 0.0;

    // Compute mult
    for (e = 0; e < k; e++)
        sum += A[i*k+e] * B[e*n+j];
    
    C[i*n+j] = sum;
}
__global__ void matmult_kernel_gpu3(double *A, double *B, double *C, int m, int n, int k)
{
    // 3 attempts: right neighbor, bottom neighbor and right neighbor with stride of m/2
    // Attempt 1: BOTTOM NEIGHBOR

    int e,start_row,i,j,t,q;

    i = (blockIdx.y * blockDim.y + threadIdx.y)*TILE_SIZE_2;
    //end_row = start_row + tile_size;
    j = blockIdx.x * blockDim.x + threadIdx.x;
    
    /*for (i = start_row; i < end_row; i++)
    {*/
        if(i < m - TILE_SIZE_2 && j < n)
        {
            // Safe.
            // Init sums
            double sum_reg[TILE_SIZE_2];
            for(e = 0; e < TILE_SIZE_2; e++)
                sum_reg[e] = 0.0;
            
            for (e = 0; e < k; e++)
            {
                for(t = 0, q = i; t < TILE_SIZE_2; t++, q++)
                    sum_reg[t]+= A[q*k+e] * B[e*n+j];
                //sum += A[i*k+e] * B[e*n+j];
                //sum2 += A[(i+1)*k+e] * B[e*n+j];
            }
            
            for(t = 0, q = i; t < TILE_SIZE_2; t++, q++)
                C[q*n+j] = sum_reg[t];
        } else
        {
            // Init sums
            double sum_reg[TILE_SIZE_2];
            for(e = 0; e < TILE_SIZE_2; e++)
                sum_reg[e] = 0.0;
            
            for (e = 0; e < k; e++)
            {
                for(t = 0, q = i; t < TILE_SIZE_2; t++, q++)
                    if(i < m && j < n)
                        sum_reg[t]+= A[q*k+e] * B[e*n+j];
            }
            
            for(t = 0, q = i; t < TILE_SIZE_2; t++, q++)
                if(i < m && j < n)    
                    C[q*n+j] = sum_reg[t];
        }
    //}
}