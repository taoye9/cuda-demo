#include "gemm.h"

#include "cublas_v2.h"

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

void gpuBlasSgemm(const float * A, const float *B, float *C, int M, int N, int K, bool no_transpose){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    
    if (status != CUBLAS_STATUS_SUCCESS){    
        printf("cublas error %s\n", _cudaGetErrorEnum(status));
        return;
    }

    float alpha = 1.0, beta = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    if (no_transpose)
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             N, M, K, 
                            &alpha, B, N, 
                            A, K, 
                            &beta, C, N);
    else{
        status = cublasSgemm(handle, 
                             CUBLAS_OP_T, CUBLAS_OP_T,
                             M,
                             N, 
                             K,
                             &alpha,
                             A,
                             K,
                             B,
                             N,
                             &beta,
                             C,
                             M
                            );
    }
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
    printf("cublasSgemm (transpose: %s) kernel total elasped time: %ld ms \n", BOOL2STR( (!no_transpose) ), duration.count());
     
    if (status != CUBLAS_STATUS_SUCCESS){    
        printf("cublas error %s\n", _cudaGetErrorEnum(status));
        return;
    }


    checkCudaErr( cudaGetLastError() );
    cublasDestroy(handle);
    return;
}
