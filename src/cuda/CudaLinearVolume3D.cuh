/*
Author: Hector Piteau (hector.piteau@gmail.com)
CudaLinearVolume3D.cuh (c) 2023
Desc: Linear Volume 3D using Cuda Malloc.
Created:  2023-04-23T17:06:41.536Z
Modified: 2023-04-24T13:03:22.194Z
*/
#ifndef CUDA_LINEAR_VOLUME3D_H
#define CUDA_LINEAR_VOLUME3D_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <glm/glm.hpp>
#include <cuda_fp16.h>
#include <iostream>
#include <random>
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

//#define VOLUME_FP16
#define VOLUME_FP32
//#define VOLUME_UINT8


struct cell {
#ifdef VOLUME_FP16
    __half2 rg; /** Hergé */
    __half2 ba; /** Béa */
#elif defined VOLUME_FP32
    float4 data;

//    float4 adam_g1;
//    float4 adam_g2;
//    float4 grads;

#elif defined VOLUME_UINT8
    ushort4 data;
#endif
};

class CudaLinearVolume3D {
private:
    cell *m_hostData = nullptr;
    cell *m_gpuData = nullptr;
    ivec3 m_res = ivec3(128, 128, 128);

    size_t m_size = 0;
    size_t m_size_gpu = 0;

public:
    explicit CudaLinearVolume3D(const ivec3 &res) {
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

    ~CudaLinearVolume3D() {
        if (m_hostData != nullptr) free(m_hostData);
        if (m_gpuData != nullptr) cudaFree(m_gpuData);
    }

    CUDA_HOST inline size_t GetIndex(const ivec3 &loc) const {
        if (loc.x < 0 || loc.y < 0 || loc.z < 0 || loc.x > m_res.x || loc.y > m_res.y || loc.z > m_res.z) {
            std::cout << "Error : trying to get index out of range. " << std::endl;
            return 0;
        }
        return loc.x * (m_res.y * m_res.z) + loc.y * (m_res.z) + loc.z;
    }

    CUDA_HOST void InitZeros() {
        for (int x = 0; x < m_res.x; ++x)
            for (int y = 0; y < m_res.y; ++y)
                for (int z = 0; z < m_res.z; ++z) {
                    HSet(ivec3(x, y, z), vec4(0.0, 0.0, 0.0, 0.0));
                }
    }

    CUDA_HOST void InitStub() {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0.0f,1.0f);
        for (int x = 0; x < m_res.x; ++x)
            for (int y = 0; y < m_res.y; ++y)
                for (int z = 0; z < m_res.z; ++z) {
                    float xf = ((float) x / (float) m_res.x);
                    float yf = ((float) y / (float) m_res.y);
                    float zf = ((float) z / (float) m_res.z);
                    vec3 tmp = vec3(xf, yf, zf) - vec3(0.5f, 0.5f, 0.5f);
                    tmp = glm::abs(tmp);

                    /** SDF */
//                    float sdf = glm::max(tmp.x, glm::max(tmp.y, tmp.z)) - 0.40f;
//                    float sdf = glm::length(tmp) - 0.40f;

                    /** OPACITY */
//                    HSet(ivec3(x, y, z),vec4(abs(sin(xf * 10.0f)), yf, zf, 0.01f));
                    HSet(ivec3(x, y, z),vec4(distribution(generator),distribution(generator),distribution(generator), 0.01f));
                }
    }

    CUDA_HOST void HSet(const ivec3 &loc, const vec4 &data) {
#ifdef VOLUME_FP16
        m_hostData[GetIndex(loc)] = {
                .rg =  __floats2half2_rn(data.r, data.g),
                .ba =  __floats2half2_rn(data.b, data.a)
        };
#elif defined VOLUME_FP32
        m_hostData[GetIndex(loc)].data = make_float4(data.x, data.y, data.z, data.w);
#endif
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

#endif //CUDA_LINEAR_VOLUME3D_H