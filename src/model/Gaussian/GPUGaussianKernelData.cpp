//
// Created by hpiteau on 09/06/23.
//
#include <glm/glm.hpp>
#include <iostream>
#include "GPUGaussianKernelData.hpp"

using namespace glm;

GPUGaussianKernelData::GPUGaussianKernelData(unsigned short dimension, unsigned short size, float sigma)
        : GPUData<GaussianWeightsDescriptor>(), m_sigma(sigma), m_dimension(dimension), m_size(size) {
    switch (dimension) {
        case 2:
            GenerateGaussianWeights2D(sigma, size);
            break;
        case 3:
            GenerateGaussianWeights3D(sigma, size);
            break;
        default:
            std::cerr << "Error creating Gaussian kernel, dimension " << std::to_string(dimension)
                      << " is not supported." << std::endl;
            break;
    }
}

void GPUGaussianKernelData::GenerateGaussianWeights3D(float sigma, unsigned short size) {
    /** Free previous kernel if already allocated. */
    cudaFree(m_device->weights);
    free(m_host->weights);
    /** Allocate new memory on device to store the kernel's weights. */
    checkCudaErrors(cudaMalloc((void **) &m_device->weights, (int) (sizeof(float) * size * size * size)));
    /** Allocate host memory to write weights into. */
    m_host->weights = (float *) malloc((sizeof(float) * size * size * size));
    if (m_host->weights == nullptr) {
        std::cerr << "GaussianKernel::GPUData: Malloc error." << std::endl;
        exit(1);
    }
    m_host->size = size;
    m_host->dim = 3;

    unsigned short sizeDiv2 = floor((float) size / 2.0f);

    float tmp, sum = 0.0f;

    for (unsigned short y = -sizeDiv2; y < sizeDiv2 + 1; y++) {
        for (unsigned short x = -sizeDiv2; x < sizeDiv2 + 1; x++) {
            for (unsigned short z = -sizeDiv2; z < sizeDiv2 + 1; z++) {
                tmp = exp(-((float) x * (float) x +
                            (float) y * (float) y +
                            (float) z * (float) z) / (2 * sigma * sigma));

                m_host->weights[(y + sizeDiv2) + (x + sizeDiv2) * size + (z + sizeDiv2) * size * size] = tmp;
                sum += tmp;
            }
        }
    }
    /** Normalisation */
    for (unsigned short y = -sizeDiv2; y < sizeDiv2 + 1; y++)
        for (unsigned short x = -sizeDiv2; x < sizeDiv2 + 1; x++)
            for (unsigned short z = -sizeDiv2; z < sizeDiv2 + 1; z++)
                m_host->weights[(y + sizeDiv2) + (x + sizeDiv2) * size + (z + sizeDiv2) * size * size] /= sum;

    ToDevice();
}

void GPUGaussianKernelData::GenerateGaussianWeights2D(float sigma, unsigned short size) {
    /** Free previous kernel if already allocated. */
    cudaFree(m_device->weights);
    free(m_host->weights);
    /** Allocate new memory on device to store the kernel's weights. */
    checkCudaErrors(cudaMalloc((void **) &m_device->weights, (int) (sizeof(float) * size * size * size)));
    /** Allocate host memory to write weights into. */
    m_host->weights = (float *) malloc((sizeof(float) * size * size * size));
    if (m_host->weights == nullptr) {
        std::cerr << "GaussianKernel::GPUData: Malloc error." << std::endl;
        exit(1);
    }

    m_host->size = size;
    m_host->dim = 2;

    unsigned short sizeDiv2 = floor((float) size / 2.0f);

    float tmp, sum = 0.0f;

    for (unsigned short y = -sizeDiv2; y < sizeDiv2 + 1; y++) {
        for (unsigned short x = -sizeDiv2; x < sizeDiv2 + 1; x++) {
            tmp = exp(-((float) x * (float) x +
                        (float) y * (float) y) /
                      (2 * sigma * sigma));
            m_host->weights[(y + sizeDiv2) + (x + sizeDiv2) * size] = tmp;
            sum += tmp;
        }
    }

    /** Normalisation */
    for (unsigned short y = -sizeDiv2; y < sizeDiv2 + 1; y++)
        for (unsigned short x = -sizeDiv2; x < sizeDiv2 + 1; x++)
            m_host->weights[(y + sizeDiv2) + (x + sizeDiv2) * size] /= sum;

    ToDevice();
}

float GPUGaussianKernelData::GetSigma() const {
    return m_sigma;
}

void GPUGaussianKernelData::SetSigma(float value) {
    m_sigma = value;
    if(m_dimension == 2){
        GenerateGaussianWeights2D(m_sigma, m_size);
    }else if(m_dimension == 3){
        GenerateGaussianWeights3D(m_sigma, m_size);
    }
}

unsigned short GPUGaussianKernelData::GetSize() const {
    return m_size;
}

void GPUGaussianKernelData::SetSize(unsigned short size) {
    m_size = size;
    if(m_dimension == 2){
        GenerateGaussianWeights2D(m_sigma, m_size);
    }else if(m_dimension == 3){
        GenerateGaussianWeights3D(m_sigma, m_size);
    }
}


