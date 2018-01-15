#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jacobi.h"
#include "gauss_seidel.h"
#include "matrixlib.h"
#define DOMAIN_LENGTH 2
#define GAUSS_SEIDEL "gauss"
#define JACOBI "jacobi"

int main(int argc, char *argv[])
{
    // Variables declaration
    int N, k_max, NN;
    double *U, *f;
    double delta, d;
    char *type;

    // 1. get run time parameters --> type, N, k_max, d (threshold)
    if (argc < 5)
    {
        printf("Error: not enough parameters passed\n");
        return 0;
    }
    else
    {
        // Should include input validation check...
        type = argv[1];
        N = atoi(argv[2]);
        NN = N + 2;
        k_max = atoi(argv[3]);
        d = atof(argv[4]);
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
    if (strcmp(type, GAUSS_SEIDEL) == 0)
    {
        //printf("Gauss-Seidel: N = %d, k_max = %d, d = %f\n", N, k_max, d);
        U = gauss_seidel_iter(NN, NN, U, d, k_max, f, delta);
    }
    else if (strcmp(type, JACOBI) == 0)
    {
        //printf("Jacobi: N = %d, k_max = %d, d = %f\n", N, k_max, d);
        U = jacobi_iter(NN, NN, U, d, k_max, f, delta);
    }
    else
        printf("Error: invalid iteration type!\n");

    // 5. print results, e.g. timings, data, etc
    //print_matrix(NN, NN, U);

    // 6. de-allocate memory
    free(U);
    free(f);
    return 0;
}