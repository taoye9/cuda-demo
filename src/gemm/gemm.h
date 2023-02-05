#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <cstdio>
#include <assert.h>
#include <time.h>
#include <chrono>

#define BOOL2STR(x) ((x) ? "true" : "false")

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float*>(&(pointer))[0])

#define checkCudaErr(func)  \
{                           \
    cudaError_t e = (func); \
                            \
    if (e != cudaSuccess){  \
        printf("%s %d cuda error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    } \
}        


__forceinline__ __host__ __device__ int float_2dim(int id1, int id2, int dim2){
    return id1 * dim2 + id2;
}

void print_mat(float * C, int m, int n);

void cpuSgemm(const float * A, const float *B, float *C, int M, int N, int K);

void gpuBlasSgemm(const float * A, const float *B, float *C, int M, int N, int K, bool no_transpose=true);

bool all_close(const float *A, const float *B, const int M, const int N);

