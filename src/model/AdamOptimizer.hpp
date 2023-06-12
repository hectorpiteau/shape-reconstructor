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

using namespace glm;

class AdamOptimizer : public SceneObject {
private:
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


public:
    explicit AdamOptimizer(const ivec3& volumeResolution);
    AdamOptimizer(const AdamOptimizer&) = delete;
    ~AdamOptimizer() override = default;

    void UpdateGPUDescriptor();

    void Step();

    void SetTargetDataVolume(std::shared_ptr<Volume3D> targetVolume);

    void SetBeta(const vec2& value);
    [[nodiscard]] const vec2& GetBeta() const;
    void SetEpsilon(float value);
    [[nodiscard]] float GetEpsilon() const;
    void SetEta(float value);
    [[nodiscard]] float GetEta() const;

    void Render() override;

    std::shared_ptr<DataLoader> GetDataLoader();


};


#endif //DRTMCS_ADAM_OPTIMIZER_HPP