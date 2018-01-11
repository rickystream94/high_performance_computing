#include <stdlib.h>
#include <stdio.h>
void init_U_matrix(int m, int n, double *mat)
{
    int i, j, value_set;
    double init_value;
    for (i = 0; i < m; i++)
    {
        value_set = 0;
        // i == 0   --> BOTTOM (NEGATIVE) BORDER
        // i == m-1 --> TOP (POSITIVE) BORDER
        if (i == 0)
        {
            init_value = 0.0;
            value_set = 1;
        }
        else if (i == m - 1)
        {
            init_value = 20.0;
            value_set = 1;
        }
        else
            init_value = 0.0;
        for (j = 0; j < n; j++)
        {
            if (!value_set)
            {
                // j == 0   -->  LEFT (NEGATIVE) BORDER
                // j == n-1 -->  RIGHT (POSITIVE) BORDER
                if (j == 0)
                    init_value = 20.0;
                else if (j == n - 1)
                    init_value = 20.0;
                else
                    init_value = 0.0;
            }
            mat[i * m + j] = init_value;
        }
    }
}

void init_f_matrix(int m, int n, double *f, double delta)
{
    int i, j;
    double delta_i, delta_j, init_value;
    double x_min = 1.0, x_max = 4.0 / 3.0, y_min = 1.0 / 3.0, y_max = 2.0 / 3.0;
    for (i = 0, delta_i = 0.0; i < m; i++)
    {
        for (j = 0, delta_j = 0.0; j < n; j++)
        {
            // Check if grid point falls inside function f where f = 200
            if (delta_j >= x_min && delta_j <= x_max && delta_i >= y_min && delta_i <= y_max)
                init_value = 200.0;
            else
                init_value = 0.0;
            f[i * m + j] = init_value;

            // Increment delta on x axis
            delta_j += delta;
        }
        // Increment delta on y axis
        delta_i += delta;
    }
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

// Pretty print matrix (nice for terminal output)
/*void print_matrix(int m, int n, double *mat)
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
}*/

void print_matrix(int m, int n, double *mat)
{
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%f", mat[i * m + j]);
            if (j != n - 1)
                printf(",");
        }
        printf("\n");
    }
}