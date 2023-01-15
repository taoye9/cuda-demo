#include "gemm.h"

#define block_size_n 128
#define block_size_n 128
#define block_size_k 8


template <const int BLOCK_SIZE_M,
          const int BLOCK_SIZE_N,
          const int BLOCK_SIZE_K,
          const int THREAD_SIZE_Y,
          const int THREAD_SIZE_X>
__global__ void gemm2(float *A, float *B, float *C, 
                      int M, int N, int K){
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // data movement from global to share mem
    const int THREAD_NUM_PER_BLOCK  = blockDim.x * blockDim.y;

    // Read B copy first.
    // copy A_tile (block_size_m, block_size_k) and transpose it to share mem (bk, bm)
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW;

    for (int row = A_TILE_ROW_START; row < BLOCK_SIZE_M; row += A_TILE_ROW_STRIDE){
        for (int dx = 0; dx < 4; dx ++){
            smem_a[A_TILE_COL + dx][row] = A[flat_2dim(blockIdx.y * BLOCK_SIZE_M + row,
                                                       A_TILE_COL + dx,
                                                       K )];      
        }
    }

    // directly copy B_tile (block_size_k, block_size_m) to share mem
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
        
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW;
    __shared__ float smem_b[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    for (int row = B_TILE_ROW_START; row < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE){
        for (int offset = 0; offset < 4; offset ++){
            smem_b[row][B_TILE_COL + offset] = B[flat_2dim(row, bx * BLOCK_SIZE_N+ B_TILE_COL + offset, N)];
        }
    }        

    __syncthreads();    

}

/* A: M * K, B: K * N, C: M * N.
 *
 *
 */

void launch_gemm2(float *A, float *B, float *C, 
                      int M, int N, int K){
    assert (block_size_m % thread_size_x == 0);
    assert (block_size_n % thread_size_y == 0);
    const int bx = block_size_m / thread_size_x;
    const int by = block_size_n / thread_size_y;

}
