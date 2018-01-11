#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "matrixlib.h"
/*------------------------------------------------- jacobi_iter -----
         |  Function jacobi_iter
         |
         |  Purpose:  EXPLAIN WHAT THIS FUNCTION DOES TO SUPPORT THE CORRECT
         |      OPERATION OF THE PROGRAM, AND HOW IT DOES IT.
         |
         |  Parameters:
         |      parameter_name (IN, OUT, or IN/OUT) -- EXPLANATION OF THE
         |              PURPOSE OF THIS PARAMETER TO THE FUNCTION.
         |                      (REPEAT THIS FOR ALL FORMAL PARAMETERS OF
         |                       THIS FUNCTION.
         |                       IN = USED TO PASS DATA INTO THIS FUNCTION,
         |                       OUT = USED TO PASS DATA OUT OF THIS FUNCTION
         |                       IN/OUT = USED FOR BOTH PURPOSES.)
         |
         |  Returns:  IF THIS FUNCTION SENDS BACK A VALUE VIA THE RETURN
         |      MECHANISM, DESCRIBE THE PURPOSE OF THAT VALUE HERE.
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
        for (i = 0, sum = 0.0; i < m; i++)
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
                    sum += pow(diff, 2);
                }
            }
        } /* end of point approximization */

        // d is now re-assigned and ready to be checked for a new iteration
        // d is calculated as the Frobenius norm
        d = sqrt(sum);

        // Swap the pointers
        // Note: mat_old now points to mat_new, and we can use the memory already allocated for mat_old to store mat_new at the next iteration (all the values will be overwritten anyway!)
        temp_ptr = mat_old;
        mat_old = mat_new;
        mat_new = temp_ptr;
    }

    // Get time
    te = omp_get_wtime() - ts;

    // Calculate number of iterations/sec
    iter_sec = (double)(k) / te;
    printf("%f\n", iter_sec);

    // When breaking the loop, according to the last pointer swap, the latest updated data is pointed by mat_old!
    return mat_old;
}