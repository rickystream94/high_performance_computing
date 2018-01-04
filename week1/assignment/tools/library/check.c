#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include "matmult.c"

void init_random_matrix(int m, int n, double **mat);
void print_matrix(int m, int n, double **mat);

int main(void)
{
    double **A, **B, **C, **C_dgemm;
    int m, n, k;

    m = 3;
    n = 4;
    k = 2;

    A = malloc_2d(m, k);
    B = malloc_2d(k, n);
    C = malloc_2d(m, n);
    C_dgemm = malloc_2d(m, n);

    printf("Initializing A and B with random values (max 5)\n");
    init_random_matrix(m, k, A);
    init_random_matrix(k, n, B);
    printf("Matrix A:\n");
    print_matrix(m, k, A);
    printf("Matrix B:\n");
    print_matrix(k, n, B);
    printf("Performing matmult...\n");
    matmult_nat(m, n, k, A, B, C);

    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, m, B, k, 0, C_dgemm, m);

    printf("C (native):\n");
    print_matrix(m, n, C);

    /*printf("C (cblas):\n");
    print_matrix(m, n, C_dgemm);*/

    return 0;
}

void init_random_matrix(int m, int n, double **mat)
{
    srand(time(NULL));
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            mat[i][j] = (double)(rand() % 5);
}

void print_matrix(int m, int n, double **mat)
{
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (j == 0)
                printf("|");
            printf(" %.2f ", mat[i][j]);
            if (j == n - 1)
                printf("|");
        }
        printf("\n");
    }
}