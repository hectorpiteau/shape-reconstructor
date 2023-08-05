//
// Created by hepiteau on 24/07/23.
//

#ifndef DRTMCS_SPARSE_VOLUME3D_HPP
#define DRTMCS_SPARSE_VOLUME3D_HPP

#include "glm/glm/glm.hpp"
#include "Common.cuh"
#include "GPUData.cuh"
#include "../../view/SceneObject/SceneObject.hpp"
#include "Volume3D.h"

#define LONG_INF 0b1111111111111111111111111111111111111111111111111111111111111111

struct stageCell {

};

struct SparseVolumeStage {
    unsigned short multiplicationFactor;
    ivec3 resolution;

};

struct SparseVolumeConfiguration {
    ivec3 initialResolution;
    unsigned short stagesAmount;
    std::vector<SparseVolumeStage> stages;
};

class SparseVolume3D : public Volume3D {
private:
    SparseVolumeConfiguration m_configuration;

    ivec3 m_initialResolution = 2 * ivec3(16, 16, 24);
    ivec3 m_maxResolution;

    unsigned int m_maxDepth = 1; /** Includes: 0, 1*/

    /** STAGE 0 */
    unsigned int stage0Size;
    stage0_cell *stage0;

    /** STAGE 1 */
    float stage1InitPercentage = 1.0f;
    unsigned int stage1Size;
    unsigned int s1Used;
    stage_cell *stage1;
    bool *s1Occupied;
    unsigned int s1_p = 0; /** Stage 1 moving pointer. */

    /** DATA */
    float dataInitPercentage = 1.0f;
    unsigned int dataSize;
    unsigned int data_p = 0; /** Data moving pointer. */

    GPUData<SparseVolumeDescriptor> m_desc;

    void Initialize();

public:
//    SparseVolume3D(const SparseVolumeConfiguration& configuration);
    SparseVolume3D(const ivec3& res);
    SparseVolume3D(const SparseVolume3D&) = delete;
    ~SparseVolume3D() = default;

    void InitStub();

    void InitializeZeros();

    void Render() override;

    /**
     * Get the Sparse Volume GPU Descriptor for interacting with the data.
     * @return
     */
    GPUData<SparseVolumeDescriptor>* GetDescriptor();

    /**
     * @brief Allocates a block of 8 cells in the Stage1 Array.
     *
     * @return int The index of the allocated bloc.
     */
    unsigned int AllocateBlockS1()
    {
        auto starting_p = s1_p % stage1Size; /** Security to be sure not to have an infinity loop. */
        bool found = false;

        /** Find free place in the array. */
        do
        {
            if (!s1Occupied[s1_p])
            {
                found = true;
                break;
            }
            s1_p = (s1_p + 1) % stage1Size; /** Loop with mod. */
        } while (starting_p != s1_p);       /** If it gets back to the starting index, it's full. */

        if (found)
        {
            /** A place is found, allocate a block and return the index. */
            s1Occupied[s1_p] = true;
            for (unsigned int i = 0; i < 8; ++i)
                stage1[s1_p].indexes[i] = INF;
            return s1_p;
        }
        else
        {
            /** FULL - need to allocate more space. */
            std::cerr << "Stage1 is full - need to allocate more space." << std::endl;
            return INF;
        }
    }

    const glm::ivec3 &GetInitialResolution();

    /** ********** Volume3D ********** */
    const glm::ivec3 &GetResolution() override;
    const glm::vec3 &GetBboxMax() override;
    const glm::vec3 &GetBboxMin() override;
    void SetBBoxMin(const vec3 &bboxMin) override;
    void SetBBoxMax(const vec3 &bboxMax) override;
    BBoxDescriptor* GetBBoxGPUDescriptor() override;
    GPUData<VolumeDescriptor>* GetGPUData() override;
    /** ********** ********** ********** */
};


#endif //DRTMCS_SPARSE_VOLUME3D_HPP
