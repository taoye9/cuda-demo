#include "gemm.h"

bool all_close(const float *A, const float *B, const int M, const int N){
    bool success = true;
    for (int j = 0; j < M ; j ++)
        for (int i = 0; i < N; i ++){
            int idx = j * N + i;
            if (A[idx] != B[idx]){
                printf("value error, (%d, %d) ,%f, %f, %f\n", j, i, A[j * N + i], B[j * N + i]);
                success = false;
                return success;
            }
        }
    return success;
}   

void print_mat(float * C, int m, int n){
    for (int i = 0; i < m; i ++){
        for (int j = 0; j < n; j ++)
            printf("%4.1f ", C[i * n + j]);
        printf("\n");
    }
    printf("\n");
    return;
}


/* A: (M, K)
 * B: (K, N)
 * C: (M, N)
 * */
void cpuSgemm(const float * A, const float *B, float *C, int M, int N, int K){
    for (int j = 0; j < M; j ++){
        for (int i = 0; i < N; i ++){
            float sum = 0.0;
            for (int k = 0; k < K; k ++)
                sum += A[j * K + k] * B[k * N + i];   
            C[j * N + i] = sum;
        }
    }
    return;
}
