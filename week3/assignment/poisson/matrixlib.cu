#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
void init_U_matrix(int N, double *mat)
{
    int i, j, value_set;
    double init_value;
    for (i = 0; i < N; i++)
    {
        value_set = 0;
        // i == 0   --> BOTTOM (NEGATIVE) BORDER
        // i == N-1 --> TOP (POSITIVE) BORDER
        if (i == 0)
        {
            init_value = 0.0;
            value_set = 1;
        }
        else if (i == N - 1)
        {
            init_value = 20.0;
            value_set = 1;
        }
        else
            init_value = 0.0;
        for (j = 0; j < N; j++)
        {
            if (!value_set)
            {
                // j == 0   -->  LEFT (NEGATIVE) BORDER
                // j == N-1 -->  RIGHT (POSITIVE) BORDER
                if (j == 0)
                    init_value = 20.0;
                else if (j == N - 1)
                    init_value = 20.0;
                else
                    init_value = 0.0;
            }
            mat[i * N + j] = init_value;
        }
    }
}

void init_f_matrix(int N, double *f, double delta)
{
    int i, j;
    double delta_i, delta_j, init_value;
    double x_min = 1.0, x_max = 4.0 / 3.0, y_min = 1.0 / 3.0, y_max = 2.0 / 3.0;
    for (i = 0; i < N; i++)
    {
        delta_i = 0.0;
        for (j = 0; j < N; j++)
        {
            delta_j = 0.0;
            // Check if grid point falls inside function f where f = 200
            if (delta_j >= x_min && delta_j <= x_max && delta_i >= y_min && delta_i <= y_max)
                init_value = 200.0;
            else
                init_value = 0.0;
            f[i * N + j] = init_value;

            // Increment delta on x axis
            delta_j += delta;
        }
        // Increment delta on y axis
        delta_i += delta;
    }
}

// Pretty print matrix (nice for terminal output)
/*void print_matrix(int N, double *mat)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (j == 0)
                printf("|");
            printf(" %.2f ", mat[i * N + j]);
            if (j == N - 1)
                printf("|");
        }
        printf("\n");
    }
}*/

void print_matrix(int N, double *mat)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%f", mat[i * N + j]);
            if (j != N - 1)
                printf(",");
        }
        printf("\n");
    }
}