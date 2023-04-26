/*
Author: Hector Piteau (hector.piteau@gmail.com)
CudaLinearVolume3D.cuh (c) 2023
Desc: Linear Volume 3D using Cuda Malloc.
Created:  2023-04-23T17:06:41.536Z
Modified: 2023-04-24T13:03:22.194Z
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <glm/glm.hpp>
#include "../utils/helper_cuda.h"

using namespace glm;

struct cell
{
    unsigned char r, g, b, a;
};

class CudaLinearVolume3D
{
private:
    float *m_hostData;
    float *m_gpuData;
    ivec3 m_res;

    bool m_isInGpu;
    size_t m_size;
    size_t m_cellSize = sizeof(struct cell);

public:
    CudaLinearVolume3D(vec3 res)
    {
        m_isInGpu = false;
        m_size = res.x * res.y * res.z * sizeof(struct cell);

        checkCudaErrors(
            cudaMalloc((void **)&m_gpuData, m_size)
        );
    }

    ~CudaLinearVolume3D()
    {
        cudaFree(m_gpuData);
    }

    void InitStub()
    {
        for (int x = 0; x < m_res.x; ++x)
            for (int y = 0; y < m_res.y; ++y)
                for (int z = 0; z < m_res.z; ++z)
                    HSet(ivec3(x, y, z), x, y, z, 255);
    }

    __device__ void DSet(ivec3 loc, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
    {
        m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z] = r;
        m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 1] = g;
        m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 2] = b;
        m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 3] = a;
    }

    __device__ void DSet(ivec3 loc, cell data)
    {
        m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z] = data.r;
        m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 1] = data.g;
        m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 2] = data.b;
        m_gpuData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 3] = data.a;
    }

    __host__ void HSet(ivec3 loc, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
    {
        m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z] = r;
        m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 1] = g;
        m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 2] = b;
        m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 3] = a;
    }

    __host__ void HSet(ivec3 loc, cell data)
    {
        m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z] = data.r;
        m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 1] = data.g;
        m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 2] = data.b;
        m_hostData[loc.x * (m_res.y * m_res.z * m_cellSize) + loc.y * (m_res.z * m_cellSize) + loc.z + 3] = data.a;
    }

    void ToGPU()
    {
        checkCudaErrors(
            cudaMemcpy(m_gpuData, m_hostData, m_size, cudaMemcpyHostToDevice));
    }

    void ToHost()
    {
        checkCudaErrors(
            cudaMemcpy(m_hostData, m_gpuData, m_size, cudaMemcpyDeviceToHost));
    }
};
