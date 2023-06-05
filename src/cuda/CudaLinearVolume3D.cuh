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
    ivec3 m_res = ivec3(100, 100, 100);

    bool m_isInGpu = false;
    size_t m_size = 0;
    size_t m_cellSize = sizeof(float4);

public:
    CudaLinearVolume3D(const ivec3& res)
    {
        m_res = res;

        m_size = m_res.x * m_res.y * m_res.z * sizeof(float4);
        
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

    CUDA_HOST inline size_t GetIndex(ivec3 loc ){
        if(loc.x < 0 || loc.y < 0 || loc.z < 0 || loc.x > m_res.x || loc.y > m_res.y || loc.z > m_res.z){
            std::cout << "Error : trying to get index out of range. " << std::endl;
            return 0;
        }
        return loc.x * (m_res.y * m_res.z) + loc.y * (m_res.z) + loc.z;
    }

    CUDA_HOST void InitStub()
    {
        for (int x = 0; x < m_res.x; ++x)
            for (int y = 0; y < m_res.y; ++y)
                for (int z = 0; z < m_res.z; ++z){
                    float xf = ((float)x/(float)m_res.x);
                    float yf = ((float)y/(float)m_res.y);
                    float zf = ((float)z/(float)m_res.z);
                    vec3 tmp = vec3(xf, yf, zf) - vec3(0.5f, 0.5f, 0.5f);
                    
                    float sdf = length(tmp) - 0.40f;
                    HSet(ivec3(x, y, z), vec4( 255.0f * xf, 255.0f * yf, 255.0f * zf, sdf));
                }
    }

    CUDA_DEV void DSet(ivec3 loc, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
    {
        float4 tmp = make_float4(r,g,b,a);m_gpuData[GetIndex(loc)] = tmp;
    }

    CUDA_DEV void DSet(ivec3 loc, cell data)
    {
        float4 tmp = make_float4(data.r, data.g, data.b, data.a);
        m_gpuData[GetIndex(loc)] = tmp;
    }

    CUDA_HOST void HSet(ivec3 loc, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
    {
        float4 tmp = make_float4(r,g,b,a);
        m_hostData[GetIndex(loc)] = tmp;
    }

    CUDA_HOST void HSet(ivec3 loc, vec4 data)
    {
        float4 tmp = make_float4(data.x,data.y,data.z,data.w);
        m_hostData[GetIndex(loc)] = tmp;
    }

    CUDA_HOST void HSet(ivec3 loc, cell data)
    {
        float4 tmp = make_float4(data.r, data.g, data.b, data.a);
        m_hostData[GetIndex(loc)] = tmp;
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