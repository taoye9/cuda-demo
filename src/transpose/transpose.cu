#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

#define NUM_REPEATS 10

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms)
{
  bool passed = true;
  for (int i = 0; i < n; i++)
    if (res[i] != ref[i]) {
      printf("%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  if (passed)
    printf("latency %f ms, bandwidth %.2f\n GB/s", ms / NUM_REPEATS, 2 * n * sizeof(float) * 1e-6 * NUM_REPEATS / ms );
}

__global__ void transposeNaive(float *d_idata, float *d_odata){
    const int x = blockIdx.x * TILE_DIM + threadIdx.x;
    const int y = blockIdx.y * TILE_DIM + threadIdx.y;
    const int N = TILE_DIM * gridDim.x;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS){
        d_odata[(x) * N + y + i] = d_idata[(y + i) * N + x];
    }      
    
    return; 
}

__global__ void transposeCoalesced(float *d_idata, float *d_odata){
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    const int x = blockIdx.x * TILE_DIM + threadIdx.x;
    const int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    const int N = gridDim.x * TILE_DIM;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS){
        tile[threadIdx.y + i][threadIdx.x] = d_idata[(y + i) * N + x];
    }
    
    __syncthreads();
    
    const int new_x = blockIdx.y * TILE_DIM + threadIdx.x;
    const int new_y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS){
        d_odata[(new_y + i) * N + new_x] = tile[threadIdx.x][threadIdx.y + i];
    }   
    return;
}

int main(){
    float ms;   
    cudaEvent_t startEvent, stopEvent;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );

    const int nx = 1024, ny = 1024;
    const int mem_size = nx * ny * sizeof(float);

    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
    dim3 dimGrid((nx + TILE_DIM - 1) / TILE_DIM, (ny + TILE_DIM - 1) / TILE_DIM);
    
    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
         nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    
    float *h_idata = (float *)malloc(mem_size);
    float *h_odata = (float *)malloc(mem_size);
    float *h_baseline = (float *)malloc(mem_size);

    float *d_idata, *d_odata;
    checkCuda( cudaMalloc(&d_idata, mem_size) );
    checkCuda( cudaMalloc(&d_odata, mem_size) );

    // init host data
    for (int y = 0; y < ny; y ++)
        for (int x = 0; x < nx; x ++)
            h_idata[y * nx + x] = y * nx + x;
    
    for (int y = 0; y < ny; y ++)
        for (int x = 0; x < nx; x ++)
            h_baseline[y * nx + x] = h_idata[x * ny + y];

    checkCuda( cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice) );
    //transpose naive
    printf("%25s", "naive transpose");
    checkCuda( cudaMemset(d_odata, 0, mem_size) );
    
    // warmup
    transposeNaive<<<dimGrid, dimBlock>>>(d_idata, d_odata);
    checkCuda( cudaEventRecord(startEvent, 0) );
    for (int i = 0; i < NUM_REPEATS; i ++){
        transposeNaive<<<dimGrid, dimBlock>>>(d_idata, d_odata);
    }
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    checkCuda( cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));

    postprocess(h_baseline, h_odata, nx * ny, ms);       

      printf("%25s", "coalesced transpose");
      checkCuda( cudaMemset(d_odata, 0, mem_size) );
      // warmup
      transposeCoalesced<<<dimGrid, dimBlock>>>(d_idata, d_odata);
      checkCuda( cudaEventRecord(startEvent, 0) );
      for (int i = 0; i < NUM_REPEATS; i++)
         transposeCoalesced<<<dimGrid, dimBlock>>>(d_idata, d_odata);
      checkCuda( cudaEventRecord(stopEvent, 0) );
      checkCuda( cudaEventSynchronize(stopEvent) );
      checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
      checkCuda( cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost) );
      postprocess(h_baseline, h_odata, nx * ny, ms);

    return 0;
}
