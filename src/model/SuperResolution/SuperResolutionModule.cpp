//
// Created by hepiteau on 19/07/23.
//

#include "SuperResolutionModule.h"


SuperResolutionModule::SuperResolutionModule(unsigned int raysAmount): m_normalDistribution(0.0f, 0.4f, -1.0f, 1.0f),  m_raysAmount(raysAmount), m_shifts(), m_desc() {
    Step();
    m_desc.Host()->shifts = static_cast<vec2 *>(GPUData<SuperResolutionDescriptor>::AllocateOnDevice(
            m_raysAmount * sizeof(glm::vec2)
    ));
    m_desc.Host()->raysAmount = m_raysAmount;
    m_desc.ToDevice();
}

void SuperResolutionModule::Step() {
    m_shifts = std::vector<glm::vec2>();

    for (unsigned int i = 0; i < m_raysAmount; ++i) {
        m_shifts.push_back(glm::vec2(
                m_normalDistribution.Get(),
                m_normalDistribution.Get()));
    }
    m_desc.ToDevice();
}

GPUData<SuperResolutionDescriptor> &SuperResolutionModule::GetDescriptor() {
    return m_desc;
}
