#include <stdlib.h>
#include <stdio.h>

void init_matrix(int m, int n, double *mat, double value)
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            mat[i * m + j] = value;
}

void init_vector(int m, double *v, double value)
{
    int i;
    for (i = 0; i < m; i++)
        v[i] = value;
}

double *malloc_2d(int m, int n)
{
    if (m <= 0 || n <= 0)
        return NULL;
    double *mat = (double *)malloc(m * n * sizeof(double));
    if (mat)
        return mat;
    return NULL;
}

void print_matrix(int m, int n, double *mat)
{
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (j == 0)
                printf("|");
            printf(" %.2f ", mat[i * m + j]);
            if (j == n - 1)
                printf("|");
        }
        printf("\n");
    }
    printf("\n\n");
}

void print_vector(int m, double* v)
{
    int i;
    for (i = 0; i < m; i++)
        printf("%.3f\n", v[i]);
    printf("\n\n");
}