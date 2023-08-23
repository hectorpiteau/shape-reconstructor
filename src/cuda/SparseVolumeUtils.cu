//
// Created by hepiteau on 24/07/23.
//

#include "SparseVolumeUtils.cuh"


__global__ void sparse_volume_init_stage0(SparseVolumeDescriptor* volume){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x > volume->stage0Size) return;
    volume->stage0[x].index = INF;
//    volume->stage0[x].active = true;
}

__global__ void sparse_volume_init_stage1(SparseVolumeDescriptor* volume){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x > volume->stage1Size) return;
    for(int i=0; i<8; ++i) volume->stage1[x].indexes[i] = INF;
//    volume->stage1[x].is_leaf = false;
    volume->stage1Occupancy[x] = false;
}

__global__ void sparse_volume_init_data(SparseVolumeDescriptor* volume){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x > volume->dataSize) return;
    volume->data[x].data = make_float4(255.0, 0.0, 0.0, 255.0);
}

extern "C" void sparse_volume_initialize(GPUData<SparseVolumeDescriptor>& volume){
    /** STAGE 0 */
    {
        dim3 threads(1024,1,1);
        /** This create enough blocks to cover the whole volume,
         * may contain threads that does not have pixel's assigned. */
        dim3 blocks((volume.Host()->stage0Size + threads.x - 1) / threads.x,1, 1);

        sparse_volume_init_stage0<<<blocks, threads>>>(volume.Device());
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "(volume_resize_double) ERROR: " << cudaGetErrorString(err) << std::endl;
        }
    }

    /** STAGE 1 */
    {
        dim3 threads(1024,1,1);
        /** This create enough blocks to cover the whole volume,
         * may contain threads that does not have pixel's assigned. */
        dim3 blocks((volume.Host()->stage1Size + threads.x - 1) / threads.x,1, 1);

        sparse_volume_init_stage1<<<blocks, threads>>>(volume.Device());
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "(volume_resize_double) ERROR: " << cudaGetErrorString(err) << std::endl;
        }
    }

    /** DATA */
    {
        dim3 threads(1024,1,1);
        /** This create enough blocks to cover the whole volume,
         * may contain threads that does not have pixel's assigned. */
        dim3 blocks((volume.Host()->dataSize + threads.x - 1) / threads.x,1, 1);

        sparse_volume_init_data<<<blocks, threads>>>(volume.Device());
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "(volume_resize_double) ERROR: " << cudaGetErrorString(err) << std::endl;
        }
    }
}