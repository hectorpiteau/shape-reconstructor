//
// Created by hpiteau on 07/06/23.
//

#ifndef DRTMCS_ADAM_OPTIMIZER_HPP
#define DRTMCS_ADAM_OPTIMIZER_HPP

#include <glm/glm.hpp>
#include <memory>
#include "../view/SceneObject/SceneObject.hpp"
#include "../cuda/CudaLinearVolume3D.cuh"
#include "Volume3D.hpp"
#include "../cuda/GPUData.cuh"
#include "../cuda/Common.cuh"
#include "DataLoader/DataLoader.hpp"
#include "VolumeRenderer.hpp"
#include "Dataset/Dataset.hpp"

using namespace glm;


struct BatchResult {
    float psnr;
};

class AdamOptimizer : public SceneObject {
private:
    Scene* m_scene;
    /** Epsilon */
    float m_epsilon=1.0E-8f;
    /** Step size. */
    float m_eta=1.0E-3f;
    /** Initialize default beta values. */
    vec2 m_beta = {0.9, 0.95};
    /** Gradient grid resolution. */
    ivec3 m_res;
    std::shared_ptr<CudaLinearVolume3D> m_adamG1;
    std::shared_ptr<CudaLinearVolume3D> m_adamG2;
    /** 3D Data to optimize. */
    std::shared_ptr<Volume3D> m_target;
    std::shared_ptr<CudaLinearVolume3D> m_blurredVoxels;

    /** Adam gradients. */


    /** Adam Optimizer GPU Data Descriptor. In order to use Adam values in a CUDA Kernel. */
    GPUData<AdamOptimizerDescriptor> m_adamDescriptor;

    /** True if the optimizer is currently working. False if not running. */
    bool m_optimize = false;
    /** Amount of steps already processed. */
    size_t m_steps = 0;

    std::shared_ptr<DataLoader> m_dataLoader;

    std::shared_ptr<Dataset> m_dataset;

    /** Integration ranges. One per camera. */
    GPUData<IntegrationRangeDescriptor> m_integrationRangeDescriptor;

    /** The overlay to show on screen useful data.*/
    std::shared_ptr<OverlayPlane> m_overlay;
    /** The cuda texture used for writing data to the overlay. */
    std::shared_ptr<CudaTexture> m_cudaTex;

    bool m_integrationRangeLoaded = false;

    std::shared_ptr<VolumeRenderer> m_volumeRenderer;

public:
    explicit AdamOptimizer(Scene* scene, std::shared_ptr<Dataset> dataset, std::shared_ptr<VolumeRenderer> volumeRenderer,  const ivec3& volumeResolution);
    AdamOptimizer(const AdamOptimizer&) = delete;
    ~AdamOptimizer() override = default;

    void UpdateGPUDescriptor();

    void Step();

    void Initialize();

//    void SetTargetDataVolume(std::shared_ptr<Volume3D> targetVolume);

    void SetBeta(const vec2& value);
    [[nodiscard]] const vec2& GetBeta() const;
    void SetEpsilon(float value);
    [[nodiscard]] float GetEpsilon() const;
    void SetEta(float value);
    [[nodiscard]] float GetEta() const;
    [[nodiscard]] bool IntegrationRangeLoaded() const;

    void Optimize();

    /**
     * Get the batch size of the DataLoader.
     *
     * @return : The amount of images in the batch.
     */
    unsigned int GetBatchSize();
    /**
     * Set the batch-size of the DataLoader.
     *
     * @param batchSize : An unsigned number that represents the batch-size.
     */
    void SetBatchSize(unsigned int batchSize);

    std::shared_ptr<DataLoader> GetDataLoader();

    void Render() override;
};


#endif //DRTMCS_ADAM_OPTIMIZER_HPP
