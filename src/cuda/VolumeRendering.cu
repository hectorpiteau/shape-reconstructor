#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>

#include "VolumeRendering.cuh"


/** 2D kernel that project rays in the volume. */
__global__ void volumeRendering(cudaSurfaceObject_t surf3d, size_t depth){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    for(int i=0; i<depth; i++){
        surf3Dwrite(2.0f, surf3d, x * 4, y, i);
    }
}

extern "C"
void volume_rendering_wrapper(cudaSurfaceObject_t& surf3d, size_t width, size_t height, size_t depth){
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x, (height + threadsperBlock.y - 1) / threadsperBlock.y);

    volumeRendering<<<numBlocks, threadsperBlock>>>(surf3d, depth);
}