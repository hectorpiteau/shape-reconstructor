#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "DenseFloat32Volume.hpp"

class SphereSDF
{
public:
static void PopulateVolume(DenseFloat32Volume* volume){
    float half = ((float)volume->GetResolution()) / 2.0f;
    float radius = 0.7f;
    int res = volume->GetResolution();
    
    /** Compute the center as the volume exact center. */
    glm::vec3 center = glm::vec3(half, half, half);

    for (int i = 0; i < res; i++)
    {
        for (int j = 0; j < res; j++)
        {
            for (int k = 0; k < res; k++)
            {
                glm::vec3 tmp = glm::vec3(i, j, k);      
                volume->SetValue(i,j,k, glm::length(tmp - center) - radius);
            }
        }
    }
}
    // SphereSDF(int resolution);

    // float GetValue(int x, int y, int z);

    // ~SphereSDF();

    // const float* GetData();
private:
    // GLenum m_textureID;
    // float *m_SDFData;
    // int m_SDFXRes;
    // int m_SDFYRes;
    // int m_SDFZRes;

    // surface<void, cudaSurfaceType3D> m_surface;
    // cudaArray* m_cuArray;
    // uchar4* m_data; // unsigned char x 4 array 
    // float4* m_dataf; // float x 4 array

    // void Compute(float* target, int resX, int resY, int resZ, float radius = 0.8f);
};
