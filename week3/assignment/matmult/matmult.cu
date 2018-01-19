extern "C" {
#include <cblas.h>
}

extern "C" {
    void matmult_nat(int m, int n, int k, double *A, double *B, double *C)
    {
        // Initializing C
        int i, j;
        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
                C[i*m+j] = 0.0;

        // Perform multiplication A*B
        int t, q;
        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
                for (q = 0, t = 0; t < k && q < k; t++, q++)
                    C[i*m+j] += A[i*m+q] * B[t*k+j];
    }

    void matmult_lib(int m, int n, int k, double *A, double *B, double *C)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 0, C, n);
    }
}