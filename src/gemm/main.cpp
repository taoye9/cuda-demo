#include "gemm.h"

#include <time.h>
#include <chrono>

#define MAT_VALUE_MAX 32 
#define M 4
#define N 4
#define K 4

void init_data(float *A, float *B, bool rand_num=false){
    for (int i = 0; i < M; i ++)
        for (int j = 0; j < K; j ++){
            if (!rand_num)
                A[i * K + j] = (float)((i * K + j) % 32);
            else
                A[i * K + j] = (float)(rand() % MAT_VALUE_MAX );
        }
    for (int i = 0; i < K; i ++ )
        for (int j = 0; j < N; j ++){
            if (!rand_num)
                B[i * K + j] = (float)((i * K + j) % 32);
            else
                B[i * K + j] = (float)(rand() % MAT_VALUE_MAX );
        }
}

int main(){
    std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
    std::chrono::duration<long, std::micro> duration;   
    srand((unsigned)time(NULL));   
    float *h_a, *h_b, *h_c, *h_c_cpu;
    h_a = (float *)malloc(M * K * sizeof(float));
    h_b = (float *)malloc(N * K * sizeof(float));
    h_c = (float *)malloc(M * N * sizeof(float));
    h_c_cpu = (float *)malloc(M * N * sizeof(float));
    
    bool rand_num = false;
    init_data(h_a, h_b, rand_num);
   
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, M * K * sizeof(float));
    cudaMalloc((void **)&d_b, N * K * sizeof(float));
    cudaMalloc((void **)&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, M * K * sizeof(float), cudaMemcpyHostToDevice);
 
    /* 
    printf("********Mat A ********\n");    
    print_mat(h_a, M, K);

    printf("********Mat B ********\n");    
    print_mat(h_b, K, N);
    */
    start = std::chrono::high_resolution_clock::now();
    cpuSgemm(h_a, h_b, h_c_cpu, M, N, K);
    stop = std::chrono::high_resolution_clock::now();
    
    /*
    printf("********Mat C ********\n");    
    print_mat(h_c_cpu, M, N);
    */
 
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
    printf("Exec naive cpu gemm kernel M = %d, N = %d, K = %d \n", M, N, K);
    printf("total elasped time: %ld ms \n", duration.count());
    
    start = std::chrono::high_resolution_clock::now();
    gpuBlasSgemm(d_a, d_b, d_c, M, N, K);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
    printf("Exec cublas gemm kernel M = %d, N = %d, K = %d \n", M, N, K);
    printf("total elasped time: %ld ms \n", duration.count());
 
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    assert(all_close(h_c, h_c_cpu, M, N));
 
    return 0;
}
