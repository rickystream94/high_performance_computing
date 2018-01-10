#include <math.h>
void gauss_seidel_iter(int m, int n, double *mat, double threshold, int k_max, double *f, double delta)
{
    int k, d, i, j;
    double sum = 0.0, diff, temp;
    for (k = 0, d = 1000000; k < k_max && d < threshold; k++)
    {
        for (i = 1; i < m - 1; i++)
        {
            for (j = 1; j < n - 1; j++)
            {
                // Copy current value in temp variable
                temp = mat[i * m + j];

                // Calculate approximization
                mat[i * m + j] = (1 / 4) * (mat[i * m + (j - 1)] + mat[i * m + (j + 1)] + mat[(i + 1) * m + j] + mat[(i - 1) * m + j] + pow(delta, 2) * f[i * m + j]);
                diff = temp - mat[i * m + j;
                sum += pow(diff, 2);
            }
        }
        d = sqrt(sum);
    }
}