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
    surface<void, cudaSurfaceType3D> mySurface;

    int width = resolution, height = resolution, depth = resolution;
    cudaArray* cuArray;
    cudaExtent extent = make_cudaExtent(width, height, depth);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaMalloc3DArray(&cuArray, &channelDesc, extent);

    cudaBindSurfaceToArray(mySurface, cuArray, channelDesc);

    uchar4* h_data = new uchar4[width * height * depth];
    // Fill h_data with your data
    cudaMemcpy3DParms memcpyParams = {0};
    memcpyParams.srcPtr = make_cudaPitchedPtr(h_data, width*sizeof(uchar4), width, height);
    memcpyParams.dstArray = cuArray;
    memcpyParams.extent = extent;
    memcpyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&memcpyParams);


    glGenTextures(1, &m_textureID);
    glBindTexture(GL_TEXTURE_2D, m_textureID);

    int c = resolution / 2;
    int radius = c * 0.8f;
    glm::vec3 center = glm::vec3(c, c, c);

    for (int i = 0; i < m_SDFXRes; i++)
    {
        for (int j = 0; j < m_SDFYRes; j++)
        {
            for (int k = 0; k < m_SDFZRes; k++)
            {
                glm::vec3 tmp = glm::vec3(i, j, k);
                m_SDFData[i * resolution * resolution + j * resolution + k] = glm::length(tmp - center) - radius;
            }
        }
    }
};

SphereSDF::~SphereSDF()
{
    delete[] m_SDFData;
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