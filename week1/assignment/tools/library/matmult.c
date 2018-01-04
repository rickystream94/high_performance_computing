#include <stdlib.h>
#include <stdio.h>

// Function prototypes
void matmult_nat(int m, int n, int k, double **A, double **B, double **C);
//double **matmult_lib(int m, int n, int k, double **A, double **B, double **C);
double **malloc_2d(int m, int n);

void matmult_nat(int m, int n, int k, double **A, double **B, double **C)
{
	printf("--- Executing matmult_nat with m = %d, n = %d, k = %d ---\n", m, n, k);
	printf("Allocating memory for C\n");
	// Allocate memory for C
	C = malloc_2d(m, n);

	// Perform multiplication A*B
	printf("Performing multiplication\n");
	int i, j, t, q;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++) {
			C[i][j] = 0.0;
			//printf("i = %d, j = %d\n", i,j);
			for (q = 0, t = 0; t < k && q < k; t++, q++)
				C[i][j] += A[i][q] * B[t][j];
		}

	printf("Done!\n");
}

/*double **
matmult_lib(int m, int n, int k, double **A, double **B, double **C)
{
	//Will call dgemm()...
}*/

double **malloc_2d(int m, int n)
{
	int i;

	if (m <= 0 || n <= 0)
		return NULL;

	double **A = malloc(m * sizeof(double *));
	if (A == NULL)
		return NULL;

	A[0] = malloc(m * n * sizeof(double));
	if (A[0] == NULL)
	{
		free(A);
		return NULL;
	}
	for (i = 1; i < m; i++)
		A[i] = A[0] + i * n;
	return A;
}
