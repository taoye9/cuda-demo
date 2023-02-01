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
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // data movement from global to share mem
    const int THREAD_NUM_PER_BLOCK  = blockDim.x * blockDim.y;

    // Read B copy first.
    // copy A_tile (block_size_m, block_size_k) and transpose it to share mem (bk, bm)
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    
    // num of float per theard for one move
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK;
    float ldg_a_register[ldg_num_a];    
    
    __shared__ float smem_a[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    for (int row = A_TILE_ROW_START; row < BLOCK_SIZE_M; row += A_TILE_ROW_STRIDE){
        int ldg_index = row / A_TILE_ROW_STRIDE * 4;
        for (int dx = 0; dx < 4; dx ++){
            ldg_a_register[ldg_index + dx] = A[flat_2dim(blockIdx.y * BLOCK_SIZE_M + row,  blockIdx.x * BLOCK_SIZE_K + A_TILE_COL + dx, K)];      
        }
        
        smem_a[0][A_TILE_COL][row] = ldg_a_register[ldg_index];
        smem_a[0][A_TILE_COL + 1][row] = ldg_a_register[ldg_index + 1];
        smem_a[0][A_TILE_COL + 2][row] = ldg_a_register[ldg_index + 2];
        smem_a[0][A_TILE_COL + 3][row] = ldg_a_register[ldg_index + 3];
    }

    // directly copy B_tile (block_size_k, block_size_m) to share mem
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
    __shared__ float smem_b[BLOCK_SIZE_K][BLOCK_SIZE_N];
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE){
        for (int j = 0; j < 4; j ++){
            smem_b[0][B_TILE_ROW_START + i][B_TILE_COL + j] = B[flat_2dim(B_TILE_ROW_START + i, blockIdx.x * BLOCK_SIZE_N + B_TILE_COL + j, N)];
        }    
    } 
    
    __syncthreads();    
 
    // prefetch from share mem to register
    for (int i = 0; i < THREAD_SIZE_Y; i += 4){
        FETCH_FLOAT4(frag_a[0][i]) = FETCH_FLOAT4(smem_a[0][threadIdx.y * THREAD_SIZE_Y  + i]);
    }

    for (int i = 0; i < THREAD_SIZE_X; i += 4){
        FETCH_FLOAT4(frag_b[0][i]) = FETCH_FLOAT4(smem_b[0][threadIdx.x * THREAD_SIZE_X + i]);
    }
    
   
    // run outer loop (K / BLOCK_SIZE_K) times
    int tile_idx = 0;      
    int ldg_write_idx = 1;

    do {
        tile_idx += BLOCK_SIZE_K;
        if (tile_idx < K){
            // fetch next tile from global mem to share mem
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE){
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                #pragma unroll
                for (int j = 0; j < 4; j ++){
                 ldg_a_register[ldg_index + j] = A[flat_2dim(blockIdx.y * BLOCK_SIZE_M + A_TILE_ROW_START + i, blockIdx.x * BLOCK_SIZE_N + A_TILE_COL + dx + tile_idx, K)]
                    
                }
            }

            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE){
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                #pragma unroll
                for (int j = 0; j < 4; j ++){
                    ldg_b_register[ldg_index + j] = B[flat_2dim(B_TILE_ROW_START + tile_idx + i, blockIdx.x * BLOC_SIZE_N + B_TILE_COL, N)];
                }
            }
        }
        

        // compute share mem tile and load next share mem tile
        for (int j = 0; j < BLOCK_SIZE_K - 1; j ++){
            for (int i = 0; i < THREAD_SIZE_Y; i += 4){
                FETCH_FLOAT4(frag_a[(j + 1)  % 2][i]) = FETCH_FLOAT4(smem_a[ldg_write_idx ^ 1][j + 1][THREAD_SIZE_Y * threadIdx.y + i]);
            }

            for (int i = 0; i < THREAD_SIZE_X; i += 4){
                FETCH_FLOAT4(frag_b[(j + 1) % 2][i]) = FETCH_FLOAT4(smem_b[ldg_write_idx ^ 1][j + 1][threadIdx.x * THREAD_SIZE_X + i]);
            }

            for (int dy = 0; dy < THREAD_SIZE_Y; dy ++){
                for (int dx = 0; dx < THREAD_SIZE_X; dx ++){
                    accum[dy][dx] += frag_a[j % 2][dy] * frag_b[j % 2][dx];
                }
            }
        }


        // load next tile from register to share mem
        if (tile_idx < K){
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE){
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                smem_a[ldg_write_idx][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_register[ldg_index];
                smem_a[ldg_write_idx][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_register[ldg_index + 1];
                smem_a[ldg_write_idx][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_register[ldg_index + 2];
                smem_a[ldg_write_idx][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_register[ldg_index + 3];
            }            


            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE){
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(smem_b[ldg_write_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }       
            __syncthreads();
            ldg_write_idx ^= 1;
        }
       

        //cornor case: last share mem tile compute and load from next tile
        {
             for (int i = 0; i < THREAD_SIZE_Y; i += 4){
                FETCH_FLOAT4(frag_a[0][i]) = FETCH_FLOAT4(smem_a[ldg_write_idx ^ 1][0][THREAD_SIZE_Y * threadIdx.y + i]);
            }

            for (int i = 0; i < THREAD_SIZE_X; i += 4){
                FETCH_FLOAT4(frag_b[0][i]) = FETCH_FLOAT4(smem_b[ldg_write_idx ^ 1][0][threadIdx.x * THREAD_SIZE_X + i]);
            }

            for (int dy = 0; dy < THREAD_SIZE_Y; dy ++){
                for (int dx = 0; dx < THREAD_SIZE_X; dx ++){
                    accum[dy][dx] += frag_a[1][dy] * frag_b[1][dx];
                }
            }
        }
    } while (tile_idx < K);
    
    for (int j = 0; j < THREAD_SIZE_Y; j ++)
        for (int i = 0; i < THREAD_SIZE_X; i ++){
            C[flat_2dim(blockIdx.y * BLOCK_SIZE_Y + threadIdx.y * THREAD_SIZE_Y + j, 
                        blockIdx.x * BLOCK_SIZE_X + threadIdx.x * THREAD_SIZE_Y + i,
                        ,N)]
        }
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
