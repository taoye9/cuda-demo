#include <cuda_runtime.h>
#include <iostream>

#define MAX_THREADS 1024
#define WARP_SIZE 32

/*  Tile method:
    1. blockDim.x == blockDim.y
    

*/
__global__ void col_sum(const float * __restrict__ din, float * __restrict__ dout, const int row, const int col){
    
    __shared__ float tile[WARP_SIZE][WARP_SIZE];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_stride = col * blockDim.y;

    float sum = 0;

    if (idx < col) {
        unsigned int offset = threadIdx.y * col + idx;
        for (int r = threadIdx.y; r < row; r += blockDim.y){
            sum += din[offset];
            offset += y_stride;
        }
    }

    tile[threadIdx.x][threadIdx.y] = sum;
    __syncthreads();    

    if (threadIdx.x == 0) {
        sum = 0;
        for (int i = 0; i < blockDim.x; i ++)
            sum += tile[threadIdx.y][i];
        int pos = blockIdx.x * blockDim.x + threadIdx.y;
        if (pos < col){
            dout[pos] = sum;
        }
    }
    return;
}


void init_data(const int row, const int col, float *hin){
    for (int y = 0; y < row; y ++) 
        for (int x = 0; x < col; x ++)
            hin[y * col + x] = (float) (x % 32);
}

void print_data(const int col, float *h){
        for (int x = 0; x < col; x ++)
            printf("%d : %f\n", x, h[x]);
}


void reduce(const int row, const int col){
    float *din, *dout, *hin, *hout; 
    hin = (float*)malloc(row * col * sizeof(float));
    hout = (float*)malloc(col * sizeof(float));
    
    init_data(row, col, hin);

    cudaMalloc((void**)&din, row * col * sizeof(float));
    cudaMalloc((void**)&dout, col * sizeof(float));
     
    cudaMemcpy(din, hin, row * col * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(WARP_SIZE, WARP_SIZE);
    dim3 grid((col - 1) / WARP_SIZE + 1); //ceil(col / WARP_SIZE)

    col_sum<<<grid, block>>>(din, dout, row, col);    
   
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
 
    cudaMemcpy(hout, dout, col * sizeof(float), cudaMemcpyDeviceToHost);
    print_data(col, hout);
    
    free(hin);
    free(hout);
    cudaFree(din);
    cudaFree(dout);
}

int main(){
    std::cout << "run reduce1 (512, 1024)" << std::endl;   
    reduce(512, 1024);

    return 0;
}
