#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "matrixlib.h"
/*------------------------------------------------- jacobi_iter -----
         |  Function jacobi_iter: PARALLELIZED VERSION: no extended parallel region (BASELINE)
         *-------------------------------------------------------------------*/
double *jacobi_iter(int m, int n, double *mat, double threshold, int k_max, double *f, double delta)
{
    int k, i, j;
    double sum = 0.0, diff, d, ts, te, iter_sec;
    double *mat_new, *mat_old, *temp_ptr;
    double d_start = 1000000.0;

    // Allocate memory for the matrix where each iteration will store the old data
    // This step is necessary for the Jacobi iteration
    mat_new = malloc_2d(m, n);

    // Assign mat_old with the initial guess (k = 0 iteration)
    mat_old = mat;

    // Get starting time
    ts = omp_get_wtime();

    // Start iteration
    for (k = 0, d = d_start; k < k_max && d > threshold; k++)
    {
        sum = 0.0;
        #pragma omp parallel for default(none) shared(m, n, mat_old, mat_new, threshold, f, delta, temp_ptr) private(j, i, diff, d) reduction(+ \
                                                                                                                                      : sum)
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                // Check boundary values (should be copied in the new matrix as they are)
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1)
                    mat_new[i * m + j] = mat_old[i * m + j];
                else
                {
                    // Calculate approximization
                    mat_new[i * m + j] = (1.0 / 4.0) * (mat_old[i * m + (j - 1)] + mat_old[i * m + (j + 1)] + mat_old[(i + 1) * m + j] + mat_old[(i - 1) * m + j] + pow(delta, 2) * f[i * m + j]);

                    // No need to include this step also for the boundary points, since diff will always be 0!
                    // sum will increase only when updating non-boundary points
                    diff = mat_old[i * m + j] - mat_new[i * m + j];
                    sum += diff * diff; //pow(diff, 2);
                }
            }
        } /* end of point approximization */

        // d is now re-assigned and ready to be checked for a new iteration
        // d is calculated as the Frobenius norm
        d = sqrt(sum);

        // Swap the pointers
        // Note: mat_old now points to mat_new, and we can use the memory already allocated for mat_old to store mat_new at the next iteration (all the values will be overwritten anyway!)
        {
            temp_ptr = mat_old;
            mat_old = mat_new;
            mat_new = temp_ptr;
        } //Implicit barrier
    }     /* end of iterations loop and of parallel */

    // Get time
    te = omp_get_wtime() - ts;
    printf("%f\n", te);

    // Calculate number of iterations/sec
    //iter_sec = (double)(k) / te;
    //printf("%f\n", iter_sec);

    // When breaking the loop, according to the last pointer swap, the latest updated data is pointed by mat_old!
    return mat_old;
}