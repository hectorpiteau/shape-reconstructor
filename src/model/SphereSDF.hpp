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
};
