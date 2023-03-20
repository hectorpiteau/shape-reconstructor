#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <iostream>
#include "../utils/Utils.hpp"
#include "SphereSDF.hpp"

#include <cuda_runtime.h>
#include <surface_types.h>
#include <surface_functions.h>


SphereSDF::SphereSDF(int resolution) : m_SDFXRes(resolution), m_SDFYRes(resolution), m_SDFZRes(resolution)
{
    m_SDFData = new float[resolution * resolution * resolution];


    /** CUDA surface */ 
    int width = resolution, height = resolution, depth = resolution;

    cudaExtent extent = make_cudaExtent(width, height, depth);
    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaMalloc3DArray(&m_cuArray, &channelDesc, extent);

    cudaBindSurfaceToArray(m_surface, m_cuArray, channelDesc);

    // m_data = new uchar4[width * height * depth];
    m_dataf = new float4[width * height * depth];
    
    /** Fill data */
    Compute(m_dataf, resolution, resolution, resolution, 0.8f);
    
    cudaMemcpy3DParms memcpyParams = {0};
    // memcpyParams.srcPtr = make_cudaPitchedPtr(m_data, width*sizeof(uchar4), width, height);
    memcpyParams.srcPtr = make_cudaPitchedPtr(m_dataf, width*sizeof(float4), width, height);
    memcpyParams.dstArray = m_cuArray;
    memcpyParams.extent = extent;
    memcpyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&memcpyParams);


    glGenTextures(1, &m_textureID);
    glBindTexture(GL_TEXTURE_2D, m_textureID);

};


void SphereSDF::Compute(float* target, int resX, int resY, int resZ, float radius){
    float half = ((float)resX) / 2.0f;
    float rad = half * radius;

    /** Compute the center as the volume exact center. */
    glm::vec3 center = glm::vec3(half, half, half);

    for (int i = 0; i < resX; i++)
    {
        for (int j = 0; j < resY; j++)
        {
            for (int k = 0; k < resZ; k++)
            {
                glm::vec3 tmp = glm::vec3(i, j, k);
                target[i * resY * resZ + j * resZ + k] = glm::length(tmp - center) - radius;
            }
        }
    }
}

SphereSDF::~SphereSDF()
{
    delete[] m_SDFData;
    delete[] m_dataf;
}

const float *SphereSDF::GetData()
{
    return m_SDFData;
}

float SphereSDF::GetValue(int x, int y, int z)
{
    if(x > m_SDFXRes || y > m_SDFYRes || z > m_SDFZRes) return 0.0f;
    return m_SDFData[x * m_SDFYRes * m_SDFZRes + y * m_SDFXRes + z];
}