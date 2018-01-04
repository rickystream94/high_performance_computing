#include <stdlib.h>

// Function prototypes
double **matmult_nat(int m, int n, int k, double **A, double **B, double **C);
double **matmult_lib(int m, int n, int k, double **A, double **B, double **C);
void init_matrix(int m, int n, double **mat);
double **malloc_2d(int m, int n);
void init_matrix(int m, int n, double **mat);

double **matmult_nat(int m, int n, int k, double **A, double **B, double **C)
{
	// Allocate memory for C
	C = malloc_2d(m, n);

	// Initialize C to zero
	init_matrix(m, n, C);

	// Perform multiplication A*B
	int i, j, t, q;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			for (q = 0, t = 0; t < k && q < k; t++, q++)
				C[i][j] += A[i][q] * B[t][j];

	return C;
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

void init_matrix(int m, int n, double **mat)
{
	int i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			mat[i][j] = 0.0;
}
