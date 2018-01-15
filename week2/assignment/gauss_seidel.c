#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "matrixlib.h"
double *gauss_seidel_iter(int m, int n, double *mat, double threshold, int k_max, double *f, double delta)
{
    int k, i, j;
    double sum = 0.0, diff, d, d_start = 1000000.0, temp, ts, te, iter_sec;

    // Get starting time
    ts = omp_get_wtime();

    // Start iteration
    for (k = 0, d = d_start; k < k_max && d > threshold; k++)
    {
        for (i = 0, sum = 0.0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                // Check boundary values
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1)
                    continue;

                // Save old value before overwriting it
                temp = mat[i * m + j];

                // Calculate approximization
                mat[i * m + j] = (1.0 / 4.0) * (mat[i * m + (j - 1)] + mat[i * m + (j + 1)] + mat[(i + 1) * m + j] + mat[(i - 1) * m + j] + pow(delta, 2) * f[i * m + j]);

                // No need to include this step also for the boundary points, since diff will always be 0!
                // sum will increase only when updating non-boundary points
                diff = temp - mat[i * m + j];
                sum += pow(diff, 2);
            }
        } /* end of point approximization */

        // d is now re-assigned and ready to be checked for a new iteration
        // d is calculated as the Frobenius norm
        d = sqrt(sum);
    }

    // Get time
    te = omp_get_wtime() - ts;

    // Calculate number of iterations/sec
    //iter_sec = (double)(k) / te;
    //printf("%f\n", iter_sec);

    // Print number of iterations (check convergence speed)
    printf("%d\n", k);

    // When breaking the loop, according to the last pointer swap, the latest updated data is pointed by mat_old!
    return mat;
}