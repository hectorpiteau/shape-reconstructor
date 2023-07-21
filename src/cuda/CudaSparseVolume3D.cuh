/*
Author: Hector Piteau (hector.piteau@gmail.com)
CudaLinearVolume3D.cuh (c) 2023
Desc: Linear Volume 3D using Cuda Malloc.
Created:  2023-04-23T17:06:41.536Z
Modified: 2023-04-24T13:03:22.194Z
*/
#ifndef CUDA_SPARSE_VOLUME3D_H
#define CUDA_SPARSE_VOLUME3D_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <glm/glm.hpp>
#include <cuda_fp16.h>
#include <iostream>
#include "../utils/helper_cuda.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

using namespace glm;


struct SparseCell {

};

class CudaSparseVolume3D {
private:
    cell *m_hostData = nullptr;
    cell *m_gpuData = nullptr;
    ivec3 m_res = ivec3(128, 128, 128);

    size_t m_size = 0;
    size_t m_size_gpu = 0;

    int* m_indirection_0; /** Base resolution: 64x64x96 - max approx : 1625^3 */
    int* m_indirection_1; /** max approx : 1625^3 */

    SparseCell* m_data;

public:
    explicit CudaSparseVolume3D(const ivec3 &res) {
        m_res = res;

        m_size = m_res.x * m_res.y * m_res.z * sizeof(cell);
        m_size_gpu = m_res.x * m_res.y * m_res.z * sizeof(cell);

        /** Declare Host buffer. */
        m_hostData = (cell *) malloc(m_size);

        /** Declare Device buffer. */
        checkCudaErrors(
                cudaMalloc((void **) &m_gpuData, m_size_gpu)
        );
    }

    ~CudaSparseVolume3D() {
        if (m_hostData != nullptr) free(m_hostData);
        if (m_gpuData != nullptr) cudaFree(m_gpuData);
    }

    /**
     * @brief Copy Host Buffer into Device's buffer.
     */
    CUDA_HOST void ToGPU() {
        checkCudaErrors(
                cudaMemcpy(m_gpuData, m_hostData, m_size, cudaMemcpyHostToDevice));
    }

    /**
     * @brief Get the Device data pointer. 
     * 
     * @return float* : A pointer on the buffer allocated on the device. 
     */
    CUDA_HOSTDEV cell *GetDevicePtr() {
        return m_gpuData;
    }

    CUDA_HOSTDEV ivec3 GetResolution() {
        return m_res;
    }
};

#endif //CUDA_SPARSE_VOLUME3D_H