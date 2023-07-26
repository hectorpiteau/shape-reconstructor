//
// Created by hepiteau on 24/07/23.
//

#include <random>
#include "SparseVolume3D.hpp"
#include "SparseVolumeUtils.cuh"

SparseVolume3D::SparseVolume3D() {
    /** Alloc the initial stage fully, indexed linearly. */
    stage0Size = m_initialResolution.x * m_initialResolution.y * m_initialResolution.z;
    m_desc.Host()->stage0Size = stage0Size;
    m_desc.Host()->stage0 = (stage0_cell*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(stage0_cell) * m_desc.Host()->stage0Size);
    std::cout << "Allocate stage0: " << std::to_string(m_desc.Host()->stage0Size) << std::endl;

    /** Allocate the first indirection stage with full size. */
    stage1Size = (unsigned int)(stage0Size);
    m_desc.Host()->stage1Size = stage1Size;
    m_desc.Host()->stage1 = (stage_cell*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(stage_cell) * m_desc.Host()->stage1Size);
    m_desc.Host()->stage1_oc = (bool*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(bool) * m_desc.Host()->stage1Size);
    std::cout << "Allocate stage1: " << std::to_string(sizeof(stage_cell) * m_desc.Host()->stage1Size) << " :: " << std::to_string(sizeof(stage_cell)) << std::endl;

    /** Allocate Data buffer. */
    dataSize = (unsigned int)(stage1Size * (4*4*4));
    m_desc.Host()->dataSize = dataSize;
    m_desc.Host()->data = (cell*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(cell) * m_desc.Host()->dataSize);
    m_desc.Host()->data_oc = (bool*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(bool) * m_desc.Host()->dataSize);
    std::cout << "Allocate data: " << std::to_string(sizeof(cell) * m_desc.Host()->dataSize) << std::endl;

    m_desc.Host()->initialResolution = m_initialResolution;
    m_desc.Host()->res = ivec3(m_initialResolution.x * 4, m_initialResolution.y * 4, m_initialResolution.z * 4);
    m_desc.Host()->worldSize = vec3(2, 2, 3);
    m_desc.Host()->bboxMax = vec3(1.0, 1.2, 1.5);
    m_desc.Host()->bboxMin = vec3(-1.0, -0.8, -1.5);
    m_desc.Host()->maxDepth = m_maxDepth;

    m_desc.ToDevice();

    Initialize();
}

GPUData<SparseVolumeDescriptor> &SparseVolume3D::GetDescriptor() {
    return m_desc;
}

void SparseVolume3D::Initialize() {
//    sparse_volume_initialize(m_desc);
}

void SparseVolume3D::InitStub() {
    /** Create temp buffers for stages and data. */
    auto stage0 = (stage0_cell*) malloc(sizeof(stage0_cell) * stage0Size);
    auto stage1 = (stage_cell*) malloc(sizeof(stage_cell) * stage1Size);
    auto stage1_oc = (bool*) malloc(sizeof(bool) * stage1Size);
    auto data = (cell*) malloc(sizeof(cell) * dataSize);
    auto data_oc = (bool*) malloc(sizeof(bool) * dataSize);

    unsigned int stage1_index_ptr = 0;
    unsigned int data_index_ptr = 0;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0f,1.0f);

    for(int x = 0; x< m_initialResolution.x; ++x){
        for(int y = 0; y< m_initialResolution.y; ++y){
            for(int z = 0; z< m_initialResolution.z; ++z){
                stage0[STAGE0_INDEX(x,y,z, m_initialResolution)].index = stage1_index_ptr;
                stage0[STAGE0_INDEX(x,y,z, m_initialResolution)].active = true;

                stage1[stage1_index_ptr].is_leaf = true;
                stage1_oc[stage1_index_ptr] = true;

                for(int i=0; i<4; ++i){
                    for(int j=0; j<4; ++j){
                        for(int k=0; k<4; ++k){
                            data[data_index_ptr].data = make_float4(distribution(generator),distribution(generator),distribution(generator), 0.01f);
                            data_oc[data_index_ptr] = true;
                            stage1[stage1_index_ptr].indexes[SHIFT_INDEX_4x4x4(ivec3(i,j,k))] = data_index_ptr;
                            data_index_ptr++;
                        }
                    }
                }
                stage1_index_ptr++;
            }
        }
    }

    /** Copy data to GPU memory. */
    cudaMemcpy(m_desc.Host()->data, data, sizeof(cell) * dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->data_oc, data_oc, sizeof(bool) * dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->stage0, stage0, sizeof(stage0_cell) * stage0Size, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->stage1, stage1, sizeof(stage_cell) * stage1Size, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->stage1_oc, stage1_oc, sizeof(bool) * stage1Size, cudaMemcpyHostToDevice);

    /** Update pointers just in case. */
    m_desc.ToDevice();
}
