#include <stdio.h>
#include <helper_cuda.h>
#define NUM_THREADS 16
#define NUM_BLOCKS 16

__global__
void hello_world()
{
    // For 1D problems, the Y values will be either 0/1! But the general formulas below will still be valid
    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid = blockIdx.x  + gridDim.x  * blockIdx.y;
    int globalThreadId = blockNumInGrid * threadsPerBlock + threadNumInBlock;
    int totalNumberOfThreads = gridDim.x*gridDim.y*blockDim.x*blockDim.y;

    if(globalThreadId==100)
    {
        int *a = (int*) 0x10000;
        *a = 0;
    }

    printf("Hello world! I'm thread %d out of %d in block %d. My global thread id is %d out of %d.\n",threadNumInBlock,threadsPerBlock,blockNumInGrid, globalThreadId,totalNumberOfThreads);
}

int main(int argc,char** argv)
{
    hello_world<<<NUM_BLOCKS,NUM_THREADS>>>();
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}