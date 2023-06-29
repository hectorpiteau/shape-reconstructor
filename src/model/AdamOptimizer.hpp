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
    float m_eta=1.0E-2f;
    /** Initialize default beta values. */
    vec2 m_beta = {0.9, 0.95};
    /** Gradient grid resolution. */
    ivec3 m_res;
    std::shared_ptr<CudaLinearVolume3D> m_adamG1;
    std::shared_ptr<CudaLinearVolume3D> m_adamG2;
    /** 3D Data to optimize. */
    std::shared_ptr<Volume3D> m_target;

    std::shared_ptr<Volume3D> m_grads;

    GPUData<VolumeDescriptor> m_gradsDescriptor;

    std::shared_ptr<CudaLinearVolume3D> m_blurredVoxels;

    /** Adam gradients. */


    /** Adam Optimizer GPU Data Descriptor. In order to use Adam values in a CUDA Kernel. */
    GPUData<AdamOptimizerDescriptor> m_adamDescriptor;

    /** True if the optimizer is currently working. False if not running. */
    bool m_optimize = false;
    /** Amount of steps already processed. */
    size_t m_steps = 1;

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

    RenderMode m_renderMode = RenderMode::PREDICTED_COLOR;

    /** Losses weightings */

    float m_color0W = 1.0f;
    float m_alpha0W = 0.0f;
    float m_alphaReg0W = -2.0f;

public:
    explicit AdamOptimizer(Scene* scene, std::shared_ptr<Dataset> dataset, std::shared_ptr<VolumeRenderer> volumeRenderer,  const ivec3& volumeResolution);
    AdamOptimizer(const AdamOptimizer&) = delete;
    ~AdamOptimizer() override = default;

    void UpdateGPUDescriptor();

    void Step();

    void Initialize();

    std::shared_ptr<Volume3D> GetGradVolume();

//    void SetTargetDataVolume(std::shared_ptr<Volume3D> targetVolume);

    void SetBeta(const vec2& value);
    [[nodiscard]] const vec2& GetBeta() const;
    void SetEpsilon(float value);
    [[nodiscard]] float GetEpsilon() const;
    void SetEta(float value);
    [[nodiscard]] float GetEta() const;
    [[nodiscard]] bool IntegrationRangeLoaded() const;

    [[nodiscard]] float GetColor0W() const { return m_color0W;}
    void SetColor0W(float value) { m_color0W = value;}
    [[nodiscard]] float GetAlpha0W() const { return m_alpha0W;}
    void SetAlpha0W(float value) { m_alpha0W = value;}
    [[nodiscard]] float GetAlphaReg0W() const { return m_alphaReg0W;}
    void SetAlphaReg0W(float value)  { m_alphaReg0W = value;}

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

    void SetRenderMode(RenderMode mode);

    RenderMode GetRenderMode();
};


#endif //DRTMCS_ADAM_OPTIMIZER_HPP
