/*
Author: Hector Piteau (hector.piteau@gmail.com)
Volume.cu (c) 2023
Desc: Volume algorithms.
Created:  2023-04-13T12:33:22.433Z
Modified: 2023-05-11T22:28:51.324Z
*/

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>

#include <cuda_gl_interop.h>

#include "../utils/helper_cuda.h"
#include <device_launch_parameters.h>
#include <cmath>

#include "Volume.cuh"
#include "../model/RayCaster/Ray.h"
#include "SingleRayCaster.cuh"

#include "Utils.cuh"
#include "Common.cuh"
#include "GPUData.cuh"

using namespace glm;

//__global__ void volume_resize_double(cell* source_volume, cell* target_volume, const ivec3& source_res, const ivec3& target_res){
//__global__ void volume_resize_double(cell* source_volume, const ivec3& source_res){
//    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
//
//    if(x > source_res.x || y > source_res.y || z > source_res.z) return;
////    if(x >= target_res.x || y >= target_res.y || z >= target_res.z) return;
//
////    target_volume[VOLUME_INDEX(x, y, z, target_res)].data.x = 1.0f;
////    target_volume[VOLUME_INDEX(x, y, z, target_res)].data.y = 0.0f;
////    target_volume[VOLUME_INDEX(x, y, z, target_res)].data.z = 0.0f;
////    target_volume[VOLUME_INDEX(x, y, z, target_res)].data.w = 1.0f;
//
//    /** For the thread in source, write its value in the target volume. */
////    auto source_cell = source_volume[VOLUME_INDEX(x,y,z, source_res)];
//    auto index = VOLUME_INDEX(x,y,z, source_res);
//    source_volume[index].data = make_float4(1.0, 0.0, 0.0, 1.0);
//
////    ivec3 target_coords = ivec3(x,y,z);
////    if(target_coords.x >= target_res.x || target_coords.y >= target_res.y || target_coords.z >= target_res.z) return;
//
//    /** same coord */
////    int index = VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target_res);
////    target_volume[index].data = make_float4(source_cell.data.x, source_cell.data.y, source_cell.data.z, source_cell.data.w);
////if(x == 2 && y == 2 && z == 2){
////    debug->i = index;
////    debug->x = x;
////    debug->y = y;
////    debug->z = z;
////    debug->iv3 = target_coords;
////
////}
//
////    target_coords = ivec3(x + 1,y,z) * 2;
////    if(target_coords.x > target_res.x || target_coords.y > target_res.y || target_coords.z > target_res.z){
////        /** x+1 */
////        target_volume[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target_res)] = source_cell;
////    }
////    target_coords = ivec3(x,y,z + 1) * 2;
////    if(target_coords.x > target_res.x || target_coords.y > target_res.y || target_coords.z > target_res.z){
////        /** z+1 */
////        target_volume[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target_res)] = source_cell;
////    }
////
////    target_coords = ivec3(x + 1,y,z + 1) * 2;
////    if(target_coords.x > target_res.x || target_coords.y > target_res.y || target_coords.z > target_res.z){
////        /** x+1, z+1 */
////        target_volume[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target_res)] = source_cell;
////    }
////
////    target_coords = ivec3(x + 1, y + 1, z) * 2;
////    if(target_coords.x > target_res.x || target_coords.y > target_res.y || target_coords.z > target_res.z){
////        /** y+1, x+1 */
////        target_volume[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target_res)] = source_cell;
////    }
////
////    target_coords = ivec3(x, y + 1, z + 1) * 2;
////    if(target_coords.x > target_res.x || target_coords.y > target_res.y || target_coords.z > target_res.z){
////        /** y+1, z+1 */
////        target_volume[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target_res)] = source_cell;
////    }
////
////    target_coords = ivec3(x + 1, y + 1, z + 1) * 2;
////    if(target_coords.x > target_res.x || target_coords.y > target_res.y || target_coords.z > target_res.z){
////        /** y+1, x+1, z+1 */
////        target_volume[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target_res)] = source_cell;
////    }
//}


__global__ void volume_resize_double(DenseVolumeDescriptor *source, DenseVolumeDescriptor *target){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x > source->res.x || y > source->res.y || z > source->res.z) return;

    /** For the thread in source, write its value in the target volume. */
    auto source_index = VOLUME_INDEX(x,y,z, source->res);
    auto source_cell = source->data[source_index];

    ivec3 target_coords_src = ivec3(x,y,z) * 2;
    ivec3 target_coords = target_coords_src;
    if(target_coords.x > target->res.x || target_coords.y > target->res.y || target_coords.z > target->res.z) return;

    /** same coord */
    target->data[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target->res)].data = make_float4(source_cell.data.x, source_cell.data.y, source_cell.data.z, source_cell.data.w);

    target_coords = target_coords_src + ivec3(1,0,0);
    if(target_coords.x < target->res.x && target_coords.y < target->res.y && target_coords.z < target->res.z){
        /** x+1 */
        target->data[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target->res)].data = make_float4(source_cell.data.x, source_cell.data.y, source_cell.data.z, source_cell.data.w);
    }
    target_coords = target_coords_src + ivec3(0,0,1);;
    if(target_coords.x < target->res.x && target_coords.y < target->res.y && target_coords.z < target->res.z){
        /** z+1 */
        target->data[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target->res)].data = make_float4(source_cell.data.x, source_cell.data.y, source_cell.data.z, source_cell.data.w);
    }

    target_coords = target_coords_src + ivec3(1,0,1);;
    if(target_coords.x < target->res.x && target_coords.y < target->res.y && target_coords.z < target->res.z){
        /** x+1, z+1 */
        target->data[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target->res)].data = make_float4(source_cell.data.x, source_cell.data.y, source_cell.data.z, source_cell.data.w);
    }

    target_coords = target_coords_src + ivec3(0,1,0);;
    if(target_coords.x < target->res.x && target_coords.y < target->res.y && target_coords.z < target->res.z){
        /** y+1 */
        target->data[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target->res)].data = make_float4(source_cell.data.x, source_cell.data.y, source_cell.data.z, source_cell.data.w);
    }

    target_coords = target_coords_src + ivec3(1,1,0);;
    if(target_coords.x < target->res.x && target_coords.y < target->res.y && target_coords.z < target->res.z){
        /** y+1, x+1 */
        target->data[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target->res)].data = make_float4(source_cell.data.x, source_cell.data.y, source_cell.data.z, source_cell.data.w);
    }

    target_coords = target_coords_src + ivec3(0,1,1);;
    if(target_coords.x < target->res.x && target_coords.y < target->res.y && target_coords.z < target->res.z){
        /** y+1, z+1 */
        target->data[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target->res)].data = make_float4(source_cell.data.x, source_cell.data.y, source_cell.data.z, source_cell.data.w);
    }

    target_coords = target_coords_src + ivec3(1,1,1);;
    if(target_coords.x < target->res.x && target_coords.y < target->res.y && target_coords.z < target->res.z){
        /** y+1, x+1, z+1 */
        target->data[VOLUME_INDEX(target_coords.x,target_coords.y,target_coords.z, target->res)].data = make_float4(source_cell.data.x, source_cell.data.y, source_cell.data.z, source_cell.data.w);
    }
}

/** Free the data cells unused. */
__global__ void volume_cull_step_1(SparseVolumeDescriptor *volume){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x > volume->res.x || y > volume->res.y || z > volume->res.z) return;

    auto indexes = SparseVolumeGetDataIndex(ivec3(x,y,z), volume);

    auto cell = volume->data[indexes.data_index];

    if(cell.data.w < 0.001f){
        /** remove cell */
        volume->data_oc[indexes.data_index] = false;

        atomicAdd((float *) (&volume->occupiedVoxelCount), -1);
    }
}

__global__ void volume_cull_step_2(SparseVolumeDescriptor *volume){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x > volume->res.x || y > volume->res.y || z > volume->res.z) return;

    auto indexes = SparseVolumeGetDataIndex(ivec3(x,y,z), volume);

    auto cell = volume->data[indexes.data_index];

    if(cell.data.w < 0.001f){
        /** remove cell */
        volume->data_oc[indexes.data_index] = false;

        atomicAdd((float *) (&volume->occupiedVoxelCount), -1);
    }
}

__global__ void volume_divide(SparseVolumeDescriptor *volume){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x > volume->res.x || y > volume->res.y || z > volume->res.z) return;

    auto coords = ivec3(x,y,z);
    /** Get the index of the data cell and check that the cell exists. */
    auto index = SparseVolumeGetDataIndex(coords, volume);

    if(index.data_index != INF){
        /** If the cell exists, check that it is worth dividing it. */
        if(volume->data[index.data_index].data.w > 0.05f){
//            volume->data[index.data_index].data.x = 1.0f;
//            volume->data[index.data_index].data.y = 0.0f;
//            volume->data[index.data_index].data.z = 0.0f;
//            volume->data[index.data_index].data.w = 1.0f;

            atomicAdd((int *) (&volume->occupiedVoxelCount), 1);

            volume->data_oc[index.data_index] = false;
            volume->stage1[index.stage1_index].is_leaf = false;

            auto data_saved = volume->data[index.data_index];

            /** retrieve the last stage1-cell that points to this data-cell and make it point
             * to a new stage1-cell. */

            auto new_stage1_index = atomicAdd((int*) &volume->stage1_pt, 1);
            int cpt = 0;
            while(!volume->stage1_oc[new_stage1_index] && cpt < 1000){
                new_stage1_index = atomicAdd((int*) &volume->stage1_pt, 1);
                cpt++;
            }

            /** Set the previous pointer to the new stage-1 cell. */
            volume->stage1[index.stage1_index].indexes[index.stage1_inner_index] = new_stage1_index;

            volume->stage1[new_stage1_index].is_leaf = true;

            for(int i=0; i < (4*4*4); i++ ){
                auto new_data_index = atomicAdd((int*) &volume->data_pt, +1);
                cpt = 0;
                while(!volume->data_oc[new_data_index] && cpt < 1000 ){
                    new_data_index = atomicAdd((int*) &volume->data_pt, 1);
                    cpt++;
                }

                volume->stage1[new_stage1_index].indexes[i] = new_data_index;

            }

            /** Duplicate the data-cell by 4*4*4 by allocating space in the data buffer for this new data   */

        }


    }


//    auto cell = volume->data[index];

//
//    atomicAdd((float *) (&volume->occupiedVoxelCount), 1);
//
//    auto new_index_stage_1 = atomicAdd((&volume->stage1_pt), 1);
}

extern "C" void volume_resize_double_wrapper(GPUData<DenseVolumeDescriptor>* source, GPUData<DenseVolumeDescriptor>* target){
    dim3 threads(8,8,8);
    /** This create enough blocks to cover the whole volume,
     * may contain threads that does not have pixel's assigned. */
    dim3 blocks((source->Host()->res.x + threads.x - 1) / threads.x,
                (source->Host()->res.y + threads.y - 1) / threads.y,
                (source->Host()->res.z + threads.z - 1) / threads.z);

    volume_resize_double<<<blocks, threads>>>(source->Device(), target->Device());
    cudaDeviceSynchronize();

    std::cout << "Resize volume done. " << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(volume_resize_double) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}

extern "C" void sparse_volume_cull_wrapper(GPUData<SparseVolumeDescriptor>* volume){
    dim3 threads(8,8,8);
    /** This create enough blocks to cover the whole volume,
     * may contain threads that does not have pixel's assigned. */
    dim3 blocks((volume->Host()->res.x + threads.x - 1) / threads.x,
                (volume->Host()->res.y + threads.y - 1) / threads.y,
                (volume->Host()->res.z + threads.z - 1) / threads.z);

    volume_cull_step_1<<<blocks, threads>>>(volume->Device());
    cudaDeviceSynchronize();

    std::cout << "Resize volume done. " << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(volume_resize_double) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}

extern "C" void sparse_volume_divide_wrapper(GPUData<SparseVolumeDescriptor>* volume){
    dim3 threads(8,8,8);
    /** This create enough blocks to cover the whole volume,
     * may contain threads that does not have pixel's assigned. */
    dim3 blocks((volume->Host()->res.x + threads.x - 1) / threads.x,
                (volume->Host()->res.y + threads.y - 1) / threads.y,
                (volume->Host()->res.z + threads.z - 1) / threads.z);

    volume_divide<<<blocks, threads>>>(volume->Device());
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(volume_resize_double) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}


