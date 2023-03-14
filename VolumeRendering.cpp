#include "VolumeRendering.hpp"
#include <glm/glm.hpp>
#include <memory>
#include "SphereSDF.hpp"

VolumeRendering::VolumeRendering()
{
    m_cameraPos = glm::vec3(2.0f, 2.0f, 2.0f);
    m_cameraLookAt = glm::vec3(0.0f, 0.0f, 0.0f);
    m_cameraRight = glm::vec3(1.0f, 0.0f, 1.0f);
    // glm::vec3 m_cameraUp = glm::vec3(2.0f, 2.0f, 2.0f);

    m_screenPlaneResolutionX = 1080;
    m_screenPlaneResolutionY = 720;

    float ratio = 200.0f;
    m_screenWidth = m_screenPlaneResolutionX / ratio;
    m_screenHeight = m_screenPlaneResolutionY / ratio;

    m_screenDistFromCamera = 1.0f;

    m_marchingStepSize = 0.1f;
}

VolumeRendering::~VolumeRendering()
{
    delete[] m_field;
}

void VolumeRendering::GenerateVectorField()
{
    /** Row storing and indexing. */
    m_field = new glm::vec3[m_screenPlaneResolutionX * m_screenPlaneResolutionY];

    for (int x = 0; x < m_screenPlaneResolutionX; ++x)
    {
        for (int y = 0; y < m_screenPlaneResolutionY; ++y)
        {
            m_field[y * m_screenPlaneResolutionX + x] = glm::vec3(0, 0, 0);
        }
    }
}

void VolumeRendering::SetSDF(std::shared_ptr<SphereSDF> sdf, int xRes, int yRes, int zRes, glm::vec3 center, glm::vec3 bbox)
{
    m_SDF = sdf;
    m_SDFXRes = xRes;
    m_SDFYRes = yRes;
    m_SDFZRes = zRes;
    m_SDFCenter = center;
    m_SDFWorldDims = bbox;

    m_SDFBboxMin = center - bbox / 2.0f;
    m_SDFBboxMax = center + bbox / 2.0f;

    m_SDFOrigin = m_SDFBboxMin;
}

bool VolumeRendering::MarchRay(const glm::vec3 ray, const glm::vec3 startPos)
{
    glm::vec3 pos = startPos;
    int max_iter = 10000;

    glm::vec3 inc = glm::normalize(ray) * m_marchingStepSize;
    glm::vec3 reso = m_SDFWorldDims / glm::vec3(m_SDFXRes, m_SDFYRes, m_SDFZRes);

    for (int i = 0; i < max_iter; ++i)
    {
        /** Increment the ray toward  it's direction vector. */
        pos += inc;

        /** Check if the ray's current point is inside the SDF bouding box or not. */
        if (glm::all(glm::greaterThan(pos, m_SDFBboxMin)) && glm::all(glm::lessThan(pos, m_SDFBboxMax)))
        {
            /** Convert the ray's current point world coordinates to sdf_local coordinates. */
            glm::vec3 tmp = pos - m_SDFOrigin;
            glm::vec3 sdfCoord = tmp / reso;
            sdfCoord = glm::floor(sdfCoord);

            /** Get the cell's value in the SDF */
            float value = m_SDF->GetValue(sdfCoord.x, sdfCoord.y, sdfCoord.z);
            
            if(value <= 0.0f){
                return true;
            }
        }
    }
    return false;
}