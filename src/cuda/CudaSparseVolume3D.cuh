/*
Author: Hector Piteau (hector.piteau@gmail.com)
CudaLinearVolume3D.cuh (c) 2023
Desc: Linear Volume 3D using Cuda Malloc.
Created:  2023-04-23T17:06:41.536Z
Modified: 2023-04-24T13:03:22.194Z
*/
#ifndef CUDA_SPARSE_VOLUME3D_H
#define CUDA_SPARSE_VOLUME3D_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <glm/glm.hpp>
#include <cuda_fp16.h>
#include <iostream>
#include <stdlib.h>
#include <memory>
#include <cstring>
#include "../utils/helper_cuda.h"
#include "Common.cuh"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#define STAGE0_INDEX(X, Y, Z, RES) ((X) * (RES).y*(RES).z + (Y) * (RES).z + (Z))
#define SHIFT_INDEX_2x2x2(shifts) (4 * (shifts).y + 2 * (shifts).z + (shifts).x)
#define SHIFT_INDEX_4x4x4(shifts) (16 * (shifts).x + 4 * (shifts).y + (shifts).z)

struct Stage0Cell
{
    unsigned int index;
};

struct StageCell
{
    unsigned int indexes[4 * 4 * 4];
    bool leafs[4 * 4 * 4];
};

struct SparseVolumeStage {
    ivec3 res;
    ivec3 cellSize;

    size_t cellsAllocatedAmount;
    size_t cellsUsedAmount;

    size_t memorySize;
};

struct SparseVolumeStage0 : public SparseVolumeStage {
    Stage0Cell* data;
};

struct SparseVolumeStage1 : public SparseVolumeStage {
    StageCell* data;
    bool* occupancy;
};

struct SparseVolumeConfiguration {
    SparseVolumeStage0 stage0;
    SparseVolumeStage1 stage1;
};


//CUDA_DEV void SparseVolumeGetDataIndex(ivec3 coords, SparseVolumeDescriptor *volume) {
//
//        /** Locate the coarser cell in the stage0. */
//        auto s0_tmp = vec3(coords) / vec3(volume->initialResolution);
//        auto s0_coords = ivec3(floor(s0_tmp));
//        auto s0_cell = volume->stage0[STAGE0_INDEX(s0_coords, volume->initialResolution)];
//
//        /** If the cell is not active then nothing is inside. return. */
//        if(!s0_cell.active) return INF;
//
//        /** If the voxel is not empty, locate the cell in the stage1. */
//        auto s1_cell = stage1[s0_cell.index];
//
//        auto previous_res = volume->initialResolution;
//
//        /** While its not a leaf, traverse the tree. */
//        while(!s1_cell.is_leaf) {
//            auto current_coords = vec3(coords) / vec3(previous_res);
//            auto s1_tmp = floor((round(current_coords) - floor(current_coords)) * 4.0f);
//            auto index = s1_cell.indexes[SHIFT_INDEX_4x4x4(ivec3(s1_tmp))];
//            s1_cell = stage1[index];
//            previous_res = previous_res * 4;
//        }
//
//        /** The cell is a leaf so the indexes correspond to the data buffer now. */
//        /** Let find the last voxel in the last block of 4x4x4. */
//        auto current_coords = vec3(coords) / vec3(previous_res);
//        auto s1_tmp = floor((round(current_coords) - floor(current_coords)) * 4.0f);
//        auto index = s1_cell.indexes[SHIFT_INDEX_4x4x4(ivec3(s1_tmp))];
//
//        return index;
//}

//CUDA_DEV inline void SparseVolumeSet(vec3 coords, vec4 value, SparseVolumeDescriptor *volume)
//{
//    auto index = SparseVolumeGetDataIndex(coords);
//    if(index == INF) return;
//    volume->data[index].data = make_float4(value.x, value.y, value.z, value.w);
//}
//
//CUDA_DEV inline void SparseVolumeAtomicSet(vec3 coords, vec4 value, SparseVolumeDescriptor *volume)
//{
//    auto index = SparseVolumeGetDataIndex(coords);
//    if(index == INF) return;
//    volume->data[index].data = make_float4(value.x, value.y, value.z, value.w);
//}
//
//CUDA_DEV inline cell SparseVolumeGet(ivec3 coords, SparseVolumeDescriptor *volume)
//{
//    auto index = SparseVolumeGetDataIndex(coords);
//    /** Check if it points to infinity, then there is no data. */
//    if(index == INF) return {.data = make_float4(0.0, 0.0, 0.0, 0.0)};
//    return volume->data[index];
//}

#endif //CUDA_SPARSE_VOLUME3D_H