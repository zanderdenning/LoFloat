
void GEMM(float* A, float* B, float* C, int M, int N, int K)
{
    // A, B, C are pointers to the matrices
    // M, N, K are the dimensions of the matrices
    // Implement the GEMM operation here
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0;
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}


//A is size m,n B is n,k C is m,k 
// acc1 size is decided by size of microkernel provided by user.
// acc2 size is decided by size of cache


template<typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixAcc1, typename MatrixAcc2>
void T_GEMM(MatrixA A, MatrixB B, MatrixC C, MatrixAcc1 acc1, MatrixAcc2 acc2)
{
  
    int m = A.m;
    int n = B.n;
    int k = A.n;

    int m_acc = acc1.m;
    int n_acc = acc1.n;

    int m_acc2 = acc2.m;
    int n_acc2 = acc2.n;

}