#include <math.h>
#include <stdio.h>
#include "matrixlib.h"
/*------------------------------------------------- gauss_seidel_iter -----
         |  Function gauss_seidel_iter
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
double *gauss_seidel_iter(int m, int n, double *mat, double threshold, int k_max, double *f, double delta)
{
    int k, i, j;
    double sum = 0.0, diff, d, d_start = 1000000.0, temp;

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

    // When breaking the loop, according to the last pointer swap, the latest updated data is pointed by mat_old!
    return mat;
}