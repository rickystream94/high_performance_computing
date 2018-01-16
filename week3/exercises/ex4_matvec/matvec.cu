#include <stdio.h>

__global__ void mat_x_vec_kernel(double *A, double *v, double *w, int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    double sum = 0.0;

    //if(i >= m)
      //  return;
    
    for (j = 0; j < n; j++)
    {
        sum += A[i * m + j] * v[j];
    }
    w[i] = sum;
}