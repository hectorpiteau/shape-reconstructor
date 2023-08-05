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

    bool m_active = false;

public:
    SuperResolutionModule(unsigned int raysAmount);
    ~SuperResolutionModule() = default;
    SuperResolutionModule(const SuperResolutionModule&) = delete;

    /**
     * Perform a step in the module. A step will change the offsets of the
     * super-resolution algorithm. The offsets are distributed based on a gaussian centered on 0.
     */
    void Step();

    /**
     * Is the module active or not.
     * @return True if active, false otherwise.
     */
    bool IsActive();

    /**
     * Set the active state of the module.
     * @param active : True if active, false if not.
     */
    void SetActive(bool active);

    /**
     * Get the GPUData descriptor of the super-resolution module.
     *
     * @return A reference to the GPUData object.
     */
    GPUData<SuperResolutionDescriptor>* GetDescriptor();

    std::vector<vec2>* GetShifts();

    /**
     * Get the amount of rays to send in the super-resolution module.
     * @return An unsigned int that correspond to the amount of rays sent in a pixel.
     */
    unsigned int GetRaysAmount();


};


#endif //DRTMCS_SUPER_RESOLUTION_MODULE_H
