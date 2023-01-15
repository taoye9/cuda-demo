#include <cuda_runtime.h>

__forceinline__ __host__ __device__ int float_2dim(int id1, int id2, int dim2){
    return id1 * dim2 + id2;
}
