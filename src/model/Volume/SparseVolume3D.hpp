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

class SparseVolume3D : public Volume3D {
private:
    /**
     * GPU Data of a sparse volume on host and device.
     */
    GPUData<SparseVolumeDescriptor> m_desc;
    ivec3 m_maxResolution;

    void Initialize();

public:
    SparseVolume3D();
    SparseVolume3D(const SparseVolume3D&) = delete;
    ~SparseVolume3D() = default;

    void InitStub();

    void InitializeZeros();

    void Render() override;

    void CullEmptyCells();

    /**
     * Get the Sparse Volume GPU Descriptor for interacting with the data.
     * @return
     */
    GPUData<SparseVolumeDescriptor>* GetDescriptor();

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
