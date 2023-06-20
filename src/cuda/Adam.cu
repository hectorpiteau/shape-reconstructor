//
// Created by hepiteau on 21/06/23.
//

#include "Adam.cuh"
#include "Common.cuh"



__global__ void UpdateAdam(VolumeDescriptor* target, AdamOptimizerDescriptor* adam){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x > target->res.x || y > target->res.y || z > target->res.z) return;




}