## cublas gemm
 cuBLAS是CUDA中专门用来解决线性代数运算的库. cuBLAS中能用于运算矩阵乘法的函数有4个，分别是 cublasSgemm（单精度实数）、cublasDgemm（双精度实数）、cublasCgemm（单精度复数）、cublasZgemm（双精度复数），它们的定义（在 cublas_v2.h 和 cublas_api.h 中）如下:
``` 
#define cublasSgemm cublasSgemm_v2
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2
(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc);
``` 

handle是用来管理cuBLAS上下文环境的, transa和transb代表op，用于决定是否转. lda, ldb leading dimension of two-dimensional array used to store the matrix A. 注意，m 是number of rows of matrix op(A) and C. n 是number of columns of matrix op(B) and C.

### cublas行优先矩阵乘法

因为cublas gemm 是 device side api 调用，启动cublas gemm 和普通的c++ fucntion 调用相同。cublas gemm 调用的难度在于参数矩阵的data layout-(行优先，列优先)的转换。pytorch默认行优先，cublas默认传入的矩阵以列优先进行存储。因为，我们有两种办法实现pytorch框架调用cublas gemm api。

1. 显式进行data layout的转换。虽然这种方式直观且简单，但增加了显存占用，和额外的执行开销。注意此时，计算结果C矩阵以列优先存储在内存，我们还需要一次显示的data layout转换。
2. 利用数学性质：
   lemma: $D \in (M \times K)$ 是含有 $M \times K$个元素的连续内存。行优先的视图 `A = D.view(M, K)` 是 列优先视图`A' = D.view(K, M)`的转置矩阵。　 
   为了计算 $C = op(A) \times op(B)$ . A, B 矩阵以行优先存储。$A \in (M, K), B \in (K, N)$ .
   
   lemma: $$C = (C^T)^T = ((A \times B)^T)^T = (B^T \times A^T)^T$$  
    
   利用lemma， 我们在cublas gemm中计算 $C^T$ 。第一个传入的参数`&B`是一个 $(N \ times K)$ 的连续内存，cublas理解为按照列优先的 $(N, K)$ 的二维矩阵。 第二个传入的参数`&A`是一个 $(K \times M)$ 的连续内存，cublas应该把B理解为按照列优先的(K, M) 的二维矩阵。cublas gemm会按照列优先方式写入计算结果 $C^T \in (N, M)$ 。pytorch框架会按照行优先理解为 $C \in (M, N)$。 

    


