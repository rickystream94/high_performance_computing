#include <stdio.h>
#include "jacobi.h"
#define KERNEL_1 1
#define KERNEL_2 2

int main(int argc, char *argv[])
{
    // Variables declaration
    int N, NN, k_max, kernel;

    // 1. get run time parameters --> type, N, k_max
    if (argc < 4)
    {
        printf("Error: not enough parameters passed\n");
        return 0;
    }
    else
    {
        // Should include input validation check...
        kernel = atoi(argv[1]);
        N = atoi(argv[2]);
        NN = N + 2;
        k_max = atoi(argv[3]);
    }

    // Call proper jacobi methods depending on chosen kernel
    switch(kernel)
    {
        case KERNEL_1:
            jacobi_1(NN, k_max);
            break;
        case KERNEL_2:
            jacobi_2(NN, k_max);
            break;
        default:
            printf("Unknown kernel type!\n");
            break;
    }

    return 0;
}