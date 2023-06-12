//
// Created by hpiteau on 07/06/23.
//

#include <memory>
#include <glm/glm.hpp>
#include <utility>
#include "AdamOptimizer.hpp"
#include "../cuda/CudaLinearVolume3D.cuh"
#include "Volume3D.hpp"

using namespace glm;

AdamOptimizer::AdamOptimizer(const ivec3 &volumeResolution) :
        SceneObject{std::string("ADAMOPTIMIZER"), SceneObjectTypes::ADAMOPTIMIZER}, m_res(volumeResolution) {
    SetName("Adam Optimizer");
    m_adamG1 = std::make_shared<CudaLinearVolume3D>(volumeResolution);
    m_adamG2 = std::make_shared<CudaLinearVolume3D>(volumeResolution);
    m_blurredVoxels = std::make_shared<CudaLinearVolume3D>(volumeResolution);
}

void AdamOptimizer::SetTargetDataVolume(std::shared_ptr<Volume3D> targetVolume){
    m_target = std::move(targetVolume);
}

void AdamOptimizer::Render() {
    //nothing for now.
    if(m_optimize){
        Step();
    }
}

void AdamOptimizer::Step(){
    /**  */
}

void AdamOptimizer::UpdateGPUDescriptor() {
    m_adamDescriptor.Host()->epsilon = m_epsilon;
    m_adamDescriptor.Host()->eta = m_eta;
    m_adamDescriptor.Host()->adamG1 = m_adamG1->GetDevicePtr();
    m_adamDescriptor.Host()->adamG2 = m_adamG1->GetDevicePtr();
    m_adamDescriptor.Host()->target = m_target->GetCudaVolume()->GetDevicePtr();
}

void AdamOptimizer::SetBeta(const vec2& value) {
    m_beta = value;
}

const vec2& AdamOptimizer::GetBeta() const {
    return m_beta;
}

void AdamOptimizer::SetEpsilon(float value) {
    m_epsilon = value;
}

float AdamOptimizer::GetEpsilon() const {
    return m_epsilon;
}

void AdamOptimizer::SetEta(float value) {
    m_eta = value;
}

float AdamOptimizer::GetEta() const {
    return m_eta;
}
