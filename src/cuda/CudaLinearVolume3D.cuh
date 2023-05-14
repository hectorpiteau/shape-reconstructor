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

struct cell
{
    unsigned char r, g, b, a;
};

class CudaLinearVolume3D
{
private:
    float4 *m_hostData = nullptr;
    float4 *m_gpuData  = nullptr;
    ivec3 m_res;

    bool m_isInGpu;
    size_t m_size;
    size_t m_cellSize = sizeof(unsigned char) * 4;

public:
    CudaLinearVolume3D(vec3 res)
    {
        m_isInGpu = false;
        m_size = res.x * res.y * res.z * 4 * sizeof(float);
        
        /** Declare Host buffer. */
        m_hostData = (float4*)malloc(m_size);

        /** Declare Device buffer. */
        checkCudaErrors(
            cudaMalloc((void **)&m_gpuData, m_size)
        );
    }

    ~CudaLinearVolume3D()
    {
        if(m_hostData != nullptr) free(m_hostData);
        if(m_gpuData != nullptr) cudaFree(m_gpuData);
    }

    CUDA_HOST void InitStub()
    {
        for (int x = 0; x < m_res.x; ++x)
            for (int y = 0; y < m_res.y; ++y)
                for (int z = 0; z < m_res.z; ++z)
                    HSet(ivec3(x, y, z), x, y, z, 255);
    }

    CUDA_DEV void DSet(ivec3 loc, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
    {
        float4 tmp = make_float4(r,g,b,a);
        m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z] = tmp;
        // m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 1] = g;
        // m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 2] = b;
        // m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 3] = a;
    }

    CUDA_DEV void DSet(ivec3 loc, cell data)
    {
        float4 tmp = make_float4(data.r, data.g, data.b, data.a);
        m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z] = tmp;
        // m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 1] = data.g;
        // m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 2] = data.b;
        // m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 3] = data.a;
    }

    CUDA_HOST void HSet(ivec3 loc, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
    {
        float4 tmp = make_float4(r,g,b,a);
        m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z] = tmp;
        // m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 1] = g;
        // m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 2] = b;
        // m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 3] = a;
    }

    CUDA_HOST void HSet(ivec3 loc, cell data)
    {
        float4 tmp = make_float4(data.r, data.g, data.b, data.a);
        m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z] = tmp;
        // m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 1] = data.g;
        // m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 2] = data.b;
        // m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 3] = data.a;
    }

    /**
     * @brief Copy Host Buffer into Device's buffer.
     */
    CUDA_HOST void ToGPU()
    {
        checkCudaErrors(
            cudaMemcpy(m_gpuData, m_hostData, m_size, cudaMemcpyHostToDevice));
    }

    /**
     * @brief Copy Device Buffer into Host's buffer.
     */
    CUDA_HOST void ToHost()
    {
        checkCudaErrors(
            cudaMemcpy(m_hostData, m_gpuData, m_size, cudaMemcpyDeviceToHost));
    }

    /**
     * @brief Get the Host data pointer. 
     * 
     * @return float* : A pointer on the buffer allocated on the host. 
     */
    CUDA_HOST float4* GetHostPtr(){
        return m_hostData;
    }

    /**
     * @brief Get the Device data pointer. 
     * 
     * @return float* : A pointer on the buffer allocated on the device. 
     */
    CUDA_HOSTDEV float4* GetDevicePtr(){
        return m_gpuData;
    }

    CUDA_HOSTDEV ivec3 GetResolution(){
        return m_res;
    }
};

#endif //CUDA_LINEAR_VOLUME3D_H