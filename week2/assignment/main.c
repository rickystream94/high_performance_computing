#include <stdio.h>
#include <stdlib.h>
#include "jacobi.h"
//#include "gauss_seidel.h"
#include "matrixlib.h"
#define DOMAIN_LENGTH 2

int main(int argc, char *argv[])
{
    // Variables declaration
    int N, k_max, NN;
    double *U, *f;
    double delta, d;

    // 1. get run time parameters --> N, k_max, threshold d
    if (argc < 4)
    {
        printf("Error: not enough parameters passed\n");
        return 0;
    }
    else
    {
        // Should include input validation check...
        N = atoi(argv[1]);
        NN = N + 2;
        k_max = atoi(argv[2]);
        d = atof(argv[3]);
    }

    // 2. allocate memory for the necessary data fields

    U = malloc_2d(NN, NN);
    f = malloc_2d(NN, NN);

    // Calculate delta
    delta = (double)(DOMAIN_LENGTH) / (NN - 1);

    // 3. initialize the fields with your start and boundary conditions
    init_f_matrix(NN, NN, f, delta);
    init_U_matrix(NN, NN, U);

    // 4. call iterator (Jacobi or Gauss-Seidel)
    printf("Jacobi: N = %d, k_max = %d, d = %f\n", N, k_max, d);
    U = jacobi_iter(NN, NN, U, d, k_max, f, delta);
    print_matrix(NN, NN, U);

    // 5. print results, e.g. timings, data, etc

    // 6. de-allocate memory
    free(U);
    free(f);
    return 0;
}