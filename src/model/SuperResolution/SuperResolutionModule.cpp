//
// Created by hepiteau on 19/07/23.
//

#include "SuperResolutionModule.h"


SuperResolutionModule::SuperResolutionModule(unsigned int raysAmount): m_normalDistribution(0.0f, 0.4f, -0.5f, 0.5f),  m_raysAmount(raysAmount), m_shifts(), m_desc() {

    m_desc.Host()->shifts = static_cast<vec2 *>(GPUData<SuperResolutionDescriptor>::AllocateOnDevice(
            m_raysAmount * sizeof(glm::vec2)
    ));

    for (unsigned int i = 0; i < m_raysAmount; ++i) {
        m_shifts.push_back(glm::vec2(
                m_normalDistribution.Get(),
                m_normalDistribution.Get()));
    }

    m_desc.Host()->raysAmount = m_raysAmount;
    m_desc.ToDevice();

}

void SuperResolutionModule::Step() {
    if(!m_active) return;
    m_shifts = std::vector<glm::vec2>();

    for (unsigned int i = 0; i < m_raysAmount; ++i) {
        m_shifts.push_back(glm::vec2(
                m_normalDistribution.Get(),
                m_normalDistribution.Get()));
    }

    checkCudaErrors(
        cudaMemcpy(m_desc.Host()->shifts, &m_shifts[0], m_raysAmount * sizeof(vec2), cudaMemcpyHostToDevice)
    );

    m_desc.ToDevice();
}

GPUData<SuperResolutionDescriptor> &SuperResolutionModule::GetDescriptor() {
    return m_desc;
}

bool SuperResolutionModule::IsActive() {
    return m_active;
}

void SuperResolutionModule::SetActive(bool active) {
    m_active = active;
    m_desc.Host()->active = active;
    m_desc.ToDevice();
}

std::vector<vec2>* SuperResolutionModule::GetShifts() {
    return &m_shifts;
}

unsigned int SuperResolutionModule::GetRaysAmount() {
    return m_raysAmount;
}
