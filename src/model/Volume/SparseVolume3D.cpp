//
// Created by hepiteau on 24/07/23.
//

#include <random>
#include "SparseVolume3D.hpp"
#include "SparseVolumeUtils.cuh"
#include "Utils.cuh"



SparseVolume3D::SparseVolume3D() {
    /** ********** SceneObject ********** */
    SetName("SparseVolume3D");
    SetTypeName("SPARSEVOLUME3D");
    SetType(SceneObjectTypes::SPARSEVOLUME3D);
    /** ********** ********** ********** */

    /** Alloc the initial stage fully, indexed linearly. */

    auto initialResolution = 2 * ivec3(32, 32, 48);

    m_desc.Host()->stage0Res = initialResolution;
    m_desc.Host()->stage0CellSize = ivec3(1, 1, 1);
    m_desc.Host()->stage0Size = m_desc.Host()->stage0Res.x * m_desc.Host()->stage0Res.y * m_desc.Host()->stage0Res.z;
    m_desc.Host()->stage0 = (Stage0Cell*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(Stage0Cell) * m_desc.Host()->stage0Size);
    std::cout << "Allocate stage0: " << std::to_string(m_desc.Host()->stage0Size) << std::endl;

    /** Allocate the first indirection stage with full size. */
    m_desc.Host()->stage1Res = m_desc.Host()->stage0Res * m_desc.Host()->stage0CellSize;
    m_desc.Host()->stage1Size = m_desc.Host()->stage1Res.x * m_desc.Host()->stage1Res.y * m_desc.Host()->stage1Res.z;
    m_desc.Host()->stage1CellSize = ivec3(4);
    m_desc.Host()->stage1 = (StageCell*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(StageCell) * m_desc.Host()->stage1Size);
    m_desc.Host()->stage1Occupancy = (bool*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(bool) * m_desc.Host()->stage1Size);
    std::cout << "Allocate stage1: " << std::to_string(sizeof(StageCell) * m_desc.Host()->stage1Size) << " :: " << std::to_string(sizeof(StageCell)) << std::endl;

    /** Allocate Data buffer. */
    m_desc.Host()->dataSize = m_desc.Host()->stage1Size * ( m_desc.Host()->stage1CellSize.x * m_desc.Host()->stage1CellSize.y * m_desc.Host()->stage1CellSize.z);
    m_desc.Host()->data = (cell*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(cell) * m_desc.Host()->dataSize);
    m_desc.Host()->data_oc = (bool*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(bool) * m_desc.Host()->dataSize);
    m_desc.Host()->data_pt = 0;

    std::cout << "Allocate data: " << std::to_string(sizeof(cell) * m_desc.Host()->dataSize) << std::endl;

    /** Other buffers */
//    m_desc.Host()->g1 = (cell*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(cell) * m_desc.Host()->dataSize);
//    m_desc.Host()->g2 = (cell*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(cell) * m_desc.Host()->dataSize);
//    m_desc.Host()->grads = (cell*) GPUData<SparseVolumeDescriptor>::AllocateOnDevice(sizeof(cell) * m_desc.Host()->dataSize);

    m_maxResolution = initialResolution * ivec3(4);

    m_desc.Host()->res = m_maxResolution;
    m_desc.Host()->worldSize = vec3(2, 2, 3);
    m_desc.Host()->bboxMax = vec3(1.0, 1.2, 1.5);
    m_desc.Host()->bboxMin = vec3(-1.0, -0.8, -1.5);

    m_desc.ToDevice();

    Initialize();
}

GPUData<SparseVolumeDescriptor>* SparseVolume3D::GetDescriptor() {
    return &m_desc;
}

void SparseVolume3D::Initialize() {

}

void SparseVolume3D::InitStub() {
    /** Create temp buffers for stages and data. */
    auto stage0 = (Stage0Cell*) malloc(sizeof(Stage0Cell) * m_desc.Host()->stage0Size);
    auto stage1 = (StageCell*) malloc(sizeof(StageCell) * m_desc.Host()->stage1Size);
    auto stage1_oc = (bool*) malloc(sizeof(bool) * m_desc.Host()->stage1Size);
    auto data = (cell*) malloc(sizeof(cell) * m_desc.Host()->dataSize);
    auto data_oc = (bool*) malloc(sizeof(bool) * m_desc.Host()->dataSize);

    unsigned int stage1_index_ptr = 0;
    unsigned int data_index_ptr = 0;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0f,1.0f);

    for(int x = 0; x < m_desc.Host()->stage0Res.x; ++x){
        for(int y = 0; y < m_desc.Host()->stage0Res.y; ++y){
            for(int z = 0; z < m_desc.Host()->stage0Res.z; ++z){

                stage0[STAGE0_INDEX(x,y,z, m_desc.Host()->stage0Res)].index = stage1_index_ptr;

                stage1_oc[stage1_index_ptr] = true;

                for(int i=0; i<4; ++i){
                    for(int j=0; j<4; ++j){
                        for(int k=0; k<4; ++k){
                            data[data_index_ptr].data = make_float4(distribution(generator),distribution(generator),distribution(generator), 0.01f);
//                            data[data_index_ptr].data = make_float4((i) * 0.25f, 0.5f, k * 0.25f, 0.1f);
                            data_oc[data_index_ptr] = true;
                            stage1[stage1_index_ptr].indexes[SHIFT_INDEX_4x4x4(ivec3(i,j,k))] = data_index_ptr;
//                            stage1[stage1_index_ptr].leafs[SHIFT_INDEX_4x4x4(ivec3(i,j,k))] = true;
                            data_index_ptr += 1;
                        }
                    }
                }
                stage1_index_ptr += 1;
            }
        }
    }

    /** Copy data to GPU memory. */
    cudaMemcpy(m_desc.Host()->data, data, sizeof(cell) * m_desc.Host()->dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->data_oc, data_oc, sizeof(bool) * m_desc.Host()->dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->stage0, stage0, sizeof(Stage0Cell) * m_desc.Host()->stage0Size, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->stage1, stage1, sizeof(StageCell) * m_desc.Host()->stage1Size, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->stage1Occupancy, stage1_oc, sizeof(bool) * m_desc.Host()->stage1Size, cudaMemcpyHostToDevice);

    /** Update pointers just in case. */
    m_desc.ToDevice();
}

void SparseVolume3D::Render() {
    /** nothing */
}

void SparseVolume3D::InitializeZeros() {
    /** Create temp buffers for stages and data. */
    auto stage0 = (Stage0Cell*) malloc(sizeof(Stage0Cell) * m_desc.Host()->stage0Size);
    auto stage1 = (StageCell*) malloc(sizeof(StageCell) * m_desc.Host()->stage1Size);
    auto stage1_oc = (bool*) malloc(sizeof(bool) *  m_desc.Host()->stage1Size);
    auto data = (cell*) malloc(sizeof(cell) *  m_desc.Host()->dataSize);
    auto data_oc = (bool*) malloc(sizeof(bool) *  m_desc.Host()->dataSize);

    unsigned int stage1_index_ptr = 0;
    unsigned int data_index_ptr = 0;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0f,1.0f);

    for(unsigned int x = 0; x < m_desc.Host()->stage0Res.x; ++x){
        for(unsigned int y = 0; y < m_desc.Host()->stage0Res.y; ++y){
            for(unsigned int z = 0; z < m_desc.Host()->stage0Res.z; ++z){
                stage0[STAGE0_INDEX(x,y,z, m_desc.Host()->stage0Res)].index = stage1_index_ptr;
                stage1_oc[stage1_index_ptr] = true;

                for(int i=0; i<4; ++i){
                    for(int j=0; j<4; ++j){
                        for(int k=0; k<4; ++k){
                            data[data_index_ptr].data = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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
    cudaMemcpy(m_desc.Host()->data, data, sizeof(cell) * m_desc.Host()->dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->data_oc, data_oc, sizeof(bool) * m_desc.Host()->dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->stage0, stage0, sizeof(Stage0Cell) * m_desc.Host()->stage0Size, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->stage1, stage1, sizeof(StageCell) * m_desc.Host()->stage1Size, cudaMemcpyHostToDevice);
    cudaMemcpy(m_desc.Host()->stage1Occupancy, stage1_oc, sizeof(bool) * m_desc.Host()->stage1Size, cudaMemcpyHostToDevice);

    /** Update pointers just in case. */
    m_desc.ToDevice();
}

const glm::ivec3 &SparseVolume3D::GetResolution() {
    return m_maxResolution;
}

const glm::vec3 &SparseVolume3D::GetBboxMax() {
    return m_desc.Host()->bboxMax;
}

const glm::vec3 &SparseVolume3D::GetBboxMin() {
    return m_desc.Host()->bboxMin;
}

void SparseVolume3D::SetBBoxMin(const vec3 &bboxMin) {
    m_desc.Host()->bboxMin = bboxMin;
    m_desc.ToDevice();
}

void SparseVolume3D::SetBBoxMax(const vec3 &bboxMax) {
    m_desc.Host()->bboxMax = bboxMax;
    m_desc.ToDevice();
}

BBoxDescriptor *SparseVolume3D::GetBBoxGPUDescriptor() {
    return nullptr;
}

GPUData<VolumeDescriptor> *SparseVolume3D::GetGPUData() {
    return (GPUData<VolumeDescriptor>*) &m_desc;
}

const glm::ivec3 &SparseVolume3D::GetInitialResolution() {
    return m_desc.Host()->stage0Res;
}

void SparseVolume3D::CullEmptyCells() {
    m_desc.ToHost();

    /** Copy data to host. */
    //TODO

    StageCell* tmpNewStage1 = (StageCell*) malloc(sizeof(StageCell) * m_desc.Host()->stage1Size);
    size_t s1Count = 0;

    cell* tmpNewData = (cell*) malloc(sizeof(cell) * m_desc.Host()->dataSize);
    size_t dataCount = 0;

    /** For every cell in stage1, check if all leafs are empty or not. */
    for(size_t i = 0; i < m_desc.Host()->stage1Size; i++){
        /** If not empty, copy to new buffer. */
        auto stage1InnerSize = m_desc.Host()->stage1CellSize.x * m_desc.Host()->stage1CellSize.y * m_desc.Host()->stage1CellSize.z;
        bool isEmpty = true;

        for(size_t j = 0; j < stage1InnerSize; ++j){
            auto index = m_desc.Host()->stage1[i].indexes[j];
            if(m_desc.Host()->data[index].data.w > 0.01f){
                isEmpty = false;
                break;
            }
        }

        if(!isEmpty){
            /** Copy the data to new buffer. */
            tmpNewStage1[s1Count] = m_desc.Host()->stage1[i];
            s1Count++;

            /** Copy data */
            for(size_t j = 0; j < stage1InnerSize; ++j){
                auto data = m_desc.Host()->data[m_desc.Host()->stage1[i].indexes[j]].data;
                tmpNewData[dataCount].data.x = data.x;
                tmpNewData[dataCount].data.y = data.y;
                tmpNewData[dataCount].data.z = data.z;
                tmpNewData[dataCount].data.w = data.w;
                dataCount++;
            }

        }

        StageCell* goodSizedStage1 = (StageCell*) malloc(sizeof(StageCell) * s1Count);
        cell* goodSizedData = (cell*) malloc(sizeof(cell) * dataCount);

        memcpy(goodSizedStage1, tmpNewStage1, sizeof(StageCell) * s1Count);
        memcpy(goodSizedData, tmpNewData, sizeof(cell) * dataCount);

        free(tmpNewStage1);
        free(tmpNewData);

        cudaFree(m_desc.Host()->stage1);
        cudaFree(m_desc.Host()->data);

        cudaMalloc(&m_desc.Host()->stage1,sizeof(StageCell) * s1Count );
        cudaMalloc(&m_desc.Host()->data,sizeof(cell) * dataCount );

        cudaMemcpy((void*) m_desc.Host()->stage1, (void*) goodSizedStage1, sizeof(StageCell) * s1Count, cudaMemcpyHostToDevice);
        cudaMemcpy((void*) m_desc.Host()->data, (void*) goodSizedData, sizeof(cell) * dataCount, cudaMemcpyHostToDevice);

        free(goodSizedStage1);
        free(goodSizedData);

        std::cout << "Cull blocks, remaining: s1: " << std::to_string(s1Count) << std::endl;
        std::cout << "Cull blocks, remaining: data: " << std::to_string(dataCount) << std::endl;
    }


    /** For every cell in a cell of stage . */

}
