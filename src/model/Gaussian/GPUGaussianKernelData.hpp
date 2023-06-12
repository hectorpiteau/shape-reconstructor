//
// Created by hpiteau on 09/06/23.
//

#ifndef DRTMCS_GAUSSIAN_KERNEL_HPP
#define DRTMCS_GAUSSIAN_KERNEL_HPP

#include <glm/glm.hpp>
#include "../../cuda/Common.cuh"
#include "../../cuda/GPUData.cuh"

using namespace glm;

class GPUGaussianKernelData : public GPUData<GaussianWeightsDescriptor> {
private:
    float m_sigma = 0.5f;
    ushort m_dimension = 2;
    ushort m_size;

    /**
     * Generate a 3D kernel of gaussian weights based on the given size and sigma.
     *
     * @param sigma : The standard deviation of the gaussian.
     * @param size : The size of the kernel, must be an odd number.
     */
    void GenerateGaussianWeights3D(float sigma, unsigned short size);

    /**
     * Generate a 2D kernel of gaussian weights based on the given size and sigma.
     *
     * @param sigma : The standard deviation of the gaussian.
     * @param size : The size of the kernel, must be an odd number.
     */
    void GenerateGaussianWeights2D(float sigma, unsigned short size);
public:
    /**
     * Construct a Gaussian kernel weight matrix ready to be imported in the GPU.
     * This class extends from the GPUData class which gives the ability to get a
     * valid pointer for cuda environment.
     *
     * @param dimension : The dimension of the kernel. Currently {2} for 2D and {3}
     * for 3D kernels are supported.
     * @param size : The size of the kernel in each direction. Must be an odd number.
     * @param sigma : The standard deviation of the gaussian kernel.
     */
    GPUGaussianKernelData(unsigned short dimension, unsigned short size, float sigma);
    GPUGaussianKernelData(const GPUGaussianKernelData &) = delete;
    ~GPUGaussianKernelData() = default;

    /**
     * Get the current kernel's standard deviation.
     * @return Standard deviation.
     */
    [[nodiscard]]float GetSigma() const;

    /**
     * Set the kernel's standard deviation and regenerate kernel's coefficients.
     * @param value : The new standard deviation.
     */
    void SetSigma(float value);

    /**
     * Set the kernel's size.
     * @return : The kernel's size.
     */
    [[nodiscard]] unsigned short GetSize() const;

    /**
    * Set the kernel's size and regenerate kernel's coefficients.
    * @param : The new size;
    */
    void SetSize(unsigned short size);
};


#endif //DRTMCS_GAUSSIAN_KERNEL_HPP
