void matmult_nat(int m, int n, int k, double *A, double *B, double *C);
void matmult_lib(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu1(int m, int n, int k, double *h_A, double *h_B, double *h_C);
void matmult_gpu2(int m, int n, int k, double *h_A, double *h_B, double *h_C);
void matmult_gpu3(int m, int n, int k, double *h_A, double *h_B, double *h_C);
void matmult_gpu4(int m, int n, int k, double *h_A, double *h_B, double *h_C);
void matmult_gpu5(int m, int n, int k, double *h_A, double *h_B, double *h_C);
void matmult_gpulib(int m, int n, int k, double *h_A, double *h_B, double *h_C);