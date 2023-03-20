#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>

class SphereSDF
{
public:
    SphereSDF(int resolution);

    float GetValue(int x, int y, int z);

    ~SphereSDF();

    const float* GetData();
private:
    GLenum m_textureID;
    float *m_SDFData;
    int m_SDFXRes;
    int m_SDFYRes;
    int m_SDFZRes;

    surface<void, cudaSurfaceType3D> m_surface;
    cudaArray* m_cuArray;
    uchar4* m_data; // unsigned char x 4 array 
    float4* m_dataf; // float x 4 array

    void Compute(float* target, int resX, int resY, int resZ, float radius = 0.8f);
};
