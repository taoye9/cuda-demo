#include <cuda_runtime.h>
#include <iostream>

__global__ void col_sum(const float * __restrict__ din, float * __restrict__ dout, const int row, const int col){
    unsigned int tid = threadIdx.x;

    float sum = 0;
    for (int i = 0; i < row; i ++)
        sum += din[i * col + tid];   
    
    dout[tid] = sum;
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


void reduce1(const int row, const int col){
    float *din, *dout, *hin, *hout; 
    hin = (float*)malloc(row * col * sizeof(float));
    hout = (float*)malloc(col * sizeof(float));
    
    init_data(row, col, hin);

    cudaMalloc((void**)&din, row * col * sizeof(float));
    cudaMalloc((void**)&dout, col * sizeof(float));
     
    cudaMemcpy(din, hin, row * col * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(col);
    dim3 grid(1);
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
    std::cout << "run reduce1 (16384, 512)" << std::endl;   
    reduce1(16384, 512);

    return 0;
}
