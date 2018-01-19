__global__ void jacobi_kernel1(int N, double *d_U_old, double *d_U_new, double *d_f, double delta);
__global__ void jacobi_kernel2(int N, double *d_U_old, double *d_U_new, double *d_f, double delta);
__global__ void jacobi_kernel_multigpu_0(int N, double *d_U_old, double *d_U_new, double *d_f, double delta);
__global__ void jacobi_kernel_multigpu_1(int N, double *d_U_old, double *d_U_new, double *d_f, double delta);