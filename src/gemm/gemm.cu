#include <cuda_runtime.h>

#include <stdio.h>

#define M 4
#define N 4
#define K 4

#define BLOCK_X 128
#define BLOCK_Y 128


#define checkCudaErr(func)  \
{                           \
    cudaError_t e = (func); \
                            \
    if (e != cudaSuccess){  \
        printf("%s %d cuda error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    } \
}        

void init_data(float * A, float *B){
    for (int i = 0; i < M; i ++)
        for (int j = 0; j < K; j ++)
            A[i * K + j] = (float)(i * K + j);

    for (int i = 0; i < K; i ++ )
        for (int j = 0; j < N; j ++)
            B[i * K + j] = (float)(i * K + j);
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

__global__ void gemm1(float *A, float *B, float *C){
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ty < M && tx < N){
        float c = 0.0;
        for (int i = 0; i < K; i ++){
            c += A[ty * K + i] * B[i * N + tx]; 
        }
        C[ty * N + tx] = c;   
    }
    
    return;
}

void launch_kernel(void (*gemm_func)(float*, float *, float *), float *a, float *b, float *c){
    unsigned int by = min(BLOCK_Y, M);
    unsigned int bx = min(BLOCK_X, N);
    
    unsigned int gy = (M + by - 1) / by;
    unsigned int gx = (N + bx - 1) / bx;
    
    dim3 grid(gx, gy);
    dim3 block(bx, by);
    printf("launch kernel gemmm <<<(%d, %d), (%d, %d)>>>\n", gx, gy, bx, by);

    gemm_func<<<grid, block>>>(a, b, c);

    return; 
}


int main(){
    float *h_a, *h_b, *h_c;
    h_a = (float *)malloc(M * K * sizeof(float));
    h_b = (float *)malloc(N * K * sizeof(float));
    h_c = (float *)malloc(M * N * sizeof(float));

    init_data(h_a, h_b);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, M * K * sizeof(float));
    cudaMalloc((void **)&d_b, N * K * sizeof(float));
    cudaMalloc((void **)&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, M * K * sizeof(float), cudaMemcpyHostToDevice);
    
    print_mat(h_a, M, K);
    print_mat(h_b, K, N);
    
    launch_kernel(gemm1, d_a, d_b, d_c);
    checkCudaErr(cudaGetLastError());
    
    cudaMemcpy(h_c, d_c, M * K * sizeof(float), cudaMemcpyDeviceToHost); 
    print_mat(h_c, M, N);
    printf("hello cuda!\n");
    cudaDeviceSynchronize(); 
}
