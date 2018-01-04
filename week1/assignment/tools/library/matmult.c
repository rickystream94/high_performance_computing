#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "matadd/datatools.c"

void init_matrix(int m, int n, double **mat, int zero)
{
	srand(time(NULL));
	int i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			if (!zero)
				mat[i][j] = (double)(rand() % 5);
			else
				mat[i][j] = 0.0;
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

int main(void)
{
	double **A, **B, **C;
	int i, j, k, q;

	int m = 3, n = 4; //to be changed dynamically
	int times = 100;

	A = malloc_2d(m, n);
	B = malloc_2d(n, m);
	C = malloc_2d(m, m);

	//Init matrix
	init_matrix(m, n, A, 0);
	init_matrix(n, m, B, 0);
	init_matrix(m, m, C, 1);

	for (i = 0; i < m; i++)
		for (j = 0; j < m; j++)
			for (k = i, q = j; k < m && q < n; k++, q++)
				C[i][j] += A[i][q] * B[k][j];

	printf("Matrix A:\n");
	print_matrix(m, n, A);
	printf("Matrix B:\n");
	print_matrix(n, m, B);
	printf("Matrix C:\n");
	print_matrix(m, m, C);
	return 0;
}
