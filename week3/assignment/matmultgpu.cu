extern "C" {
#include <stdio.h>
}
#define TILE_SIZE_2 2
#define TILE_SIZE 4
#define BLOCK_SIZE 16

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
    int e,i,j,t;
    i = (blockIdx.y * blockDim.y + threadIdx.y)*TILE_SIZE_2;
    j = (blockIdx.x * blockDim.x + threadIdx.x);
    
    //if(i < m && j < n - TILE_SIZE_2) // RIGHT NEIGHBOR VERSION
    if(i < m - TILE_SIZE_2 && j < n) // BOTTOM NEIGHBOR VERSION
    {
        // Safe
        double sum_reg[TILE_SIZE_2];
        for(e = 0; e < TILE_SIZE_2; e++)
            sum_reg[e] = 0.0;
        
        for (e = 0; e < k; e++)
        {
            //for(t = 0; t < TILE_SIZE_2; t++) // RIGHT NEIGHBOR VERSION
            for(t = 0; t < TILE_SIZE_2; t++) // BOTTOM NEIGHBOR VERSION
                sum_reg[t]+= A[(i+t)*k+e] * B[e*n+j]; // BOTTOM NEIGHBOR VERSION
                //sum_reg[t]+= A[i*k+e] * B[e*n+(j+t)]; // RIGHT NEIGHBOR VERSION
        }
        
        for(t = 0; t < TILE_SIZE_2; t++) // BOTTOM NEIGHBOR VERSION
        //for(t = 0; t < TILE_SIZE_2; t++) // RIGHT NEIGHBOR VERSION
            C[(i+t)*n+j] = sum_reg[t]; // BOTTOM NEIGHBOR VERSION
            //C[i*n+(j+t)] = sum_reg[t]; // RIGHT NEIGHBOR VERSION
    } else
    {
        // Unsafe
        double sum_reg[TILE_SIZE_2];
        for(e = 0; e < TILE_SIZE_2; e++)
            sum_reg[e] = 0.0;
        
        for (e = 0; e < k; e++)
        {
            //for(t = 0; t < TILE_SIZE_2; t++) // RIGHT NEIGHBOR VERSION
            for(t = 0; t < TILE_SIZE_2; t++) // BOTTOM NEIGHBOR VERSION
                if((i+t) < m && j < n) // BOTTOM NEIGHBOR VERSION
                //if(i < m && (j+t) < n) // RIGHT NEIGHBOR VERSION
                    sum_reg[t]+= A[(i+t)*k+e] * B[e*n+j]; // BOTTOM NEIGHBOR VERSION
                    //sum_reg[t]+= A[i*k+e] * B[e*n+(j+t)]; // RIGHT NEIGHBOR VERSION
        }
        
        for(t = 0; t < TILE_SIZE_2; t++) // BOTTOM NEIGHBOR VERSION
        //for(t = 0; t < TILE_SIZE_2; t++) // RIGHT NEIGHBOR VERSION
            if((i+t) < m && j < n) // BOTTOM NEIGHBOR VERSION
            //if(i < m && (j+t) < n) // RIGHT NEIGHBOR VERSION
                C[(i+t)*n+j] = sum_reg[t]; // BOTTOM NEIGHBOR VERSION
                //C[i*n+(j+t)] = sum_reg[t]; // RIGHT NEIGHBOR VERSION
    }
}

__global__ void matmult_kernel_gpu4(double *A, double *B, double *C, int m, int n, int k)
{
    int e,i,j,t;
    i = (blockIdx.y * blockDim.y + threadIdx.y)*TILE_SIZE;
    j = (blockIdx.x * blockDim.x + threadIdx.x);
    
    if(i < m - TILE_SIZE && j < n)
    {
        // Safe
        double sum_reg[TILE_SIZE];
        for(e = 0; e < TILE_SIZE; e++)
            sum_reg[e] = 0.0;
        
        for (e = 0; e < k; e++)
        {
            for(t = 0; t < TILE_SIZE; t++)
                sum_reg[t]+= A[(i+t)*k+e] * B[e*n+j];
        }
        
        for(t = 0; t < TILE_SIZE; t++)
            C[(i+t)*n+j] = sum_reg[t];
    } else
    {
        // Unsafe
        double sum_reg[TILE_SIZE];
        for(e = 0; e < TILE_SIZE; e++)
            sum_reg[e] = 0.0;
        
        for (e = 0; e < k; e++)
        {
            for(t = 0; t < TILE_SIZE; t++)
                if((i+t) < m && j < n)
                    sum_reg[t]+= A[(i+t)*k+e] * B[e*n+j];
        }
        
        for(t = 0; t < TILE_SIZE; t++)
            if((i+t) < m && j < n)
                C[(i+t)*n+j] = sum_reg[t];
    }
}

__global__ void matmult_kernel_gpu5(double *A, double *B, double *C, int m, int n, int k)
{
    int e,q;

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    double Cvalue = 0.0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (e = 0; e < (k / BLOCK_SIZE); ++e) {
        // Shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Each thread loads one element of each sub-matrix
        As[row][col] = A[k * blockRow * BLOCK_SIZE + e * BLOCK_SIZE + row * k + col];
        Bs[row][col] = B[n * e * BLOCK_SIZE + blockCol * BLOCK_SIZE + row * n + col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (q = 0; q < BLOCK_SIZE; ++q)
            Cvalue += As[row][q] * Bs[q][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Each thread writes one element to C in device memory 
    C[n * blockRow * BLOCK_SIZE + blockCol * BLOCK_SIZE + row * n + col] = Cvalue;
}