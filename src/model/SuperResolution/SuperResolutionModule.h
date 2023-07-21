//
// Created by hepiteau on 19/07/23.
//

#ifndef DRTMCS_SUPER_RESOLUTION_MODULE_H
#define DRTMCS_SUPER_RESOLUTION_MODULE_H

#include "../Distribution/NormalDistributionClamped.hpp"
#include "Common.cuh"
#include "GPUData.cuh"

#include <memory>
#include <glm/glm.hpp>

class SuperResolutionModule {
private:
    NormalDistributionClamped<float> m_normalDistribution;
    unsigned int m_raysAmount;

    std::vector<glm::vec2> m_shifts;

    GPUData<SuperResolutionDescriptor> m_desc;

public:
    SuperResolutionModule(unsigned int raysAmount);
    ~SuperResolutionModule() = default;
    SuperResolutionModule(const SuperResolutionModule&) = delete;

    void Step();

    GPUData<SuperResolutionDescriptor>& GetDescriptor();

};


#endif //DRTMCS_SUPER_RESOLUTION_MODULE_H
