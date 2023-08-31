//
// Created by hpiteau on 07/06/23.
//

#ifndef DRTMCS_ADAM_OPTIMIZER_HPP
#define DRTMCS_ADAM_OPTIMIZER_HPP

#include <glm/glm.hpp>
#include <memory>
#include "../view/SceneObject/SceneObject.hpp"
#include "../cuda/CudaLinearVolume3D.cuh"
#include "Volume/DenseVolume3D.hpp"
#include "../cuda/GPUData.cuh"
#include "../cuda/Common.cuh"
#include "DataLoader/DataLoader.hpp"
#include "VolumeRenderer.hpp"
#include "Dataset/Dataset.hpp"
#include "SuperResolution/SuperResolutionModule.h"
#include "Distribution/UniformDistribution.hpp"
#include "Volume/SparseVolume3D.hpp"
#include "Statistics/Statistics.h"

using namespace glm;

struct LevelOfDetail {
    short level;
    ivec3 volume_res;
    ivec2 image_res;
    std::string image_train_path;
    std::string json_train_path;
    std::string image_valid_path;
    std::string json_valid_path;
};

#define LOD_AMOUNT 4

const LevelOfDetail LODs[LOD_AMOUNT] = {
        {
                .level = 1,
                .volume_res = ivec3(32 * 2, 32 * 2, 32 * 3),
                .image_res = ivec2(200, 200),
                .image_train_path = std::string("../data/nerf200/train"),
                .json_train_path = std::string("../data/nerf/transforms_train.json"),
                .image_valid_path = std::string("../data/nerf200/val"),
                .json_valid_path = std::string("../data/nerf/transforms_valid.json")
        },
        {
                .level = 2,
                .volume_res = 2 * ivec3(32 * 2, 32 * 2, 32 * 3),
                .image_res = ivec2(400, 400),
                .image_train_path = std::string("../data/nerf200/train"),
                .json_train_path = std::string("../data/nerf/transforms_train.json"),
                .image_valid_path = std::string("../data/nerf200/val"),
                .json_valid_path = std::string("../data/nerf/transforms_valid.json")
        },
        {
                .level = 3,
                .volume_res = 4 * ivec3(32 * 2, 32 * 2, 32 * 3),
                .image_res = ivec2(800, 800),
                .image_train_path = std::string("../data/nerf200/train"),
                .json_train_path = std::string("../data/nerf/transforms_train.json"),
                .image_valid_path = std::string("../data/nerf200/val"),
                .json_valid_path = std::string("../data/nerf/transforms_valid.json")
        },
        {
                .level = 4,
                .volume_res = 8 * ivec3(32 * 2, 32 * 2, 32 * 3),
                .image_res = ivec2(800, 800),
                .image_train_path = std::string("../data/nerf200/train"),
                .json_train_path = std::string("../data/nerf/transforms_train.json"),
                .image_valid_path = std::string("../data/nerf200/val"),
                .json_valid_path = std::string("../data/nerf/transforms_valid.json")
        }
};

struct BatchResult {
    float psnr;
};

class AdamOptimizer : public SceneObject {
private:
    Scene *m_scene;
    /** Epsilon */
    float m_epsilon = 1.0E-8f;
    /** Step size. */
    float m_eta = 0.1E-2f;
    /** Initialize default beta values. */
    vec2 m_beta = {0.9, 0.95};

    /** Gradient grid resolution. */
    std::shared_ptr<DenseVolume3D> m_adamG1;
    std::shared_ptr<SparseVolume3D> m_s_adamG1;

    std::shared_ptr<DenseVolume3D> m_adamG2;
    std::shared_ptr<SparseVolume3D> m_s_adamG2;

    /** 3D Data to optimize. */
    std::shared_ptr<DenseVolume3D> m_target;
    std::shared_ptr<SparseVolume3D> m_s_target;

    /** Adam gradients. */
    std::shared_ptr<DenseVolume3D> m_grads;
    std::shared_ptr<SparseVolume3D> m_s_grads;

    GPUData<DenseVolumeDescriptor> m_gradsDescriptor;

    /** Adam Optimizer GPU Data Descriptor. In order to use Adam values in a CUDA Kernel. */
    GPUData<AdamOptimizerDescriptor> m_adamDescriptor;
    GPUData<SparseAdamOptimizerDescriptor> m_s_adamDescriptor;

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
    float m_alpha0W = 1.0f;
    float m_alphaReg0W = -1.0f;
    float m_TVL20W = 1.0f;

    /** LOD */
    uint m_currentLODIndex = 0;

    /** Super Resolution */
    SuperResolutionModule m_superResModule;
    unsigned short m_amountOfGradientsToWrite = 2;

    UniformDistribution<short> m_uniformDistribution;

    std::shared_ptr<Statistics> m_stats;

public:
    explicit AdamOptimizer(Scene *scene, std::shared_ptr<Dataset> dataset, std::shared_ptr<DenseVolume3D> target,
                           std::shared_ptr<VolumeRenderer> renderer, std::shared_ptr<SparseVolume3D> sparseVolume, std::shared_ptr<Statistics> statistics);

    AdamOptimizer(const AdamOptimizer &) = delete;

    ~AdamOptimizer() override = default;

    void UpdateGPUDescriptor();

    void Step();

    void Initialize();

    std::shared_ptr<DenseVolume3D> GetTargetVolume();

    std::shared_ptr<DenseVolume3D> GetGradVolume();

    void SetBeta(const vec2 &value);

    [[nodiscard]] const vec2 &GetBeta() const;

    void SetEpsilon(float value);

    [[nodiscard]] float GetEpsilon() const;

    void SetEta(float value);

    [[nodiscard]] float GetEta() const;

    [[nodiscard]] bool IntegrationRangeLoaded() const;

    [[nodiscard]] float GetColor0W() const { return m_color0W; }

    void SetColor0W(float value) { m_color0W = value; }

    [[nodiscard]] float GetAlpha0W() const { return m_alpha0W; }

    void SetAlpha0W(float value) { m_alpha0W = value; }

    [[nodiscard]] float GetAlphaReg0W() const { return m_alphaReg0W; }

    void SetAlphaReg0W(float value) { m_alphaReg0W = value; }

    [[nodiscard]] float GetTVL20W() const { return m_TVL20W; }

    void SetTVL20W(float value) { m_TVL20W = value; }

    /**
     * Start/Stop the optimization procedure.
     */
    void Optimize();

    /**
     * Go to the next level of details.
     * Increase the volume's resolution and image resolution too.
     */
    void NextLOD();

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

    void SetUseSuperResolution(bool value);

    bool UseSuperResolution();

    SuperResolutionModule* GetSuperResolutionModule();

    void CullVolume();

    void DivideVolume();
};


#endif //DRTMCS_ADAM_OPTIMIZER_HPP
