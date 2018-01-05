#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include <time.h>

// Function prototypes
void matmult_nat(int m, int n, int k, double **A, double **B, double **C);
void matmult_mnk(int m, int n, int k, double **A, double **B, double **C);
void matmult_nmk(int m, int n, int k, double **A, double **B, double **C);
void matmult_kmn(int m, int n, int k, double **A, double **B, double **C);
void matmult_knm(int m, int n, int k, double **A, double **B, double **C);
void matmult_mkn(int m, int n, int k, double **A, double **B, double **C);
void matmult_nkm(int m, int n, int k, double **A, double **B, double **C);
void matmult_lib(int m, int n, int k, double **A, double **B, double **C);
double **malloc_2d(int m, int n);
void print_matrix(int m, int n, double **mat);

void matmult_nat(int m, int n, int k, double **A, double **B, double **C)
{
	//printf("--- Executing matmult_nat with m = %d, n = %d, k = %d ---\n", m, n, k);

	// Initializing C
	int i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			C[i][j] = 0.0;

	// Perform multiplication A*B
	//printf("Performing multiplication\n");
	int t, q;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			for (q = 0, t = 0; t < k && q < k; t++, q++)
				C[i][j] += A[i][q] * B[t][j];

	// Checking with CBLAS
	//double **C_cblas;
	//C_cblas = malloc_2d(m, n);
	//matmult_lib(m,n,k,A,B,C);
	//printf("C (cblas):\n");
	//print_matrix(m, n, C_cblas);
	//printf("C (native):\n");
	//print_matrix(m, n, C);

	//printf("Done!\n");
}

void matmult_mnk(int m, int n, int k, double **A, double **B, double **C)
{
	// Initializing C
	int i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			C[i][j] = 0.0;

	// Perform multiplication A*B
	int t, q;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			for (q = 0, t = 0; t < k && q < k; t++, q++)
				C[i][j] += A[i][q] * B[t][j];
}

void matmult_nmk(int m, int n, int k, double **A, double **B, double **C)
{
	// Initializing C
	int i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			C[i][j] = 0.0;

	// Perform multiplication A*B
	int t, q;
	for (j = 0; j < n; j++)
		for (i = 0; i < m; i++)
			for (q = 0, t = 0; t < k && q < k; t++, q++)
				C[i][j] += A[i][q] * B[t][j];
}

void matmult_kmn(int m, int n, int k, double **A, double **B, double **C)
{
	// Initializing C
	int i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			C[i][j] = 0.0;

	// Perform multiplication A*B
	int t, q;
	for (q = 0, t = 0; t < k && q < k; t++, q++)
		for (i = 0; i < m; i++)
			for (j = 0; j < n; j++)
				C[i][j] += A[i][q] * B[t][j];
}

void matmult_knm(int m, int n, int k, double **A, double **B, double **C)
{
	// Initializing C
	int i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			C[i][j] = 0.0;

	// Perform multiplication A*B
	int t, q;
	for (q = 0, t = 0; t < k && q < k; t++, q++)
		for (j = 0; j < n; j++)
			for (i = 0; i < m; i++)
				C[i][j] += A[i][q] * B[t][j];
}

void matmult_mkn(int m, int n, int k, double **A, double **B, double **C)
{
	// Initializing C
	int i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			C[i][j] = 0.0;

	// Perform multiplication A*B
	int t, q;
	for (i = 0; i < m; i++)
		for (q = 0, t = 0; t < k && q < k; t++, q++)
			for (j = 0; j < n; j++)
				C[i][j] += A[i][q] * B[t][j];
}

void matmult_nkm(int m, int n, int k, double **A, double **B, double **C)
{
	// Initializing C
	int i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			C[i][j] = 0.0;

	// Perform multiplication A*B
	int t, q;
	for (j = 0; j < n; j++)
		for (q = 0, t = 0; t < k && q < k; t++, q++)
			for (i = 0; i < m; i++)
				C[i][j] += A[i][q] * B[t][j];
}

void matmult_lib(int m, int n, int k, double **A, double **B, double **C)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A[0], k, B[0], n, 0, C[0], n);
}

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

void print_matrix(int m, int n, double **mat)
{
	int i, j;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (j == 0)
				printf("|");
			printf(" %.2f ", mat[i][j]);
			if (j == n - 1)
				printf("|");
		}
		printf("\n");
	}
}