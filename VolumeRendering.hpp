#ifndef VOLUME_RENDERING_H
#define VOLUME_RENDERING_H
#include <glm/glm.hpp>
#include <memory>
#include "SphereSDF.hpp"


class VolumeRendering {
public:
    VolumeRendering();
    ~VolumeRendering();

    /**
     * @brief Generate a field of normalized vectors that 
     * goes from the camera toward each pixels. 
     * 
     */
    void GenerateVectorField();

    void SetSDF(std::shared_ptr<SphereSDF> sdf, int xRes, int yRes, int zRes, glm::vec3 center, glm::vec3 bbox);

    bool MarchRay(const glm::vec3 ray, const glm::vec3 startPos);

    
private:

    glm::vec3* m_field;

    glm::vec3 m_cameraPos;
    glm::vec3 m_cameraLookAt;
    glm::vec3 m_cameraRight;
    glm::vec3 m_cameraUp;

    int m_screenPlaneResolutionX;
    int m_screenPlaneResolutionY;
    
    float m_screenWidth;
    float m_screenHeight;

    float m_screenDistFromCamera;

    float m_marchingStepSize;

    /** SDF */
    std::shared_ptr<SphereSDF> m_SDF;
    int m_SDFXRes;
    int m_SDFYRes;
    int m_SDFZRes;
    glm::vec3 m_SDFCenter;
    glm::vec3 m_SDFWorldDims;

    glm::vec3 m_SDFBboxMin;
    glm::vec3 m_SDFBboxMax;

    glm::vec3 m_SDFOrigin;
};

#endif //VOLUME_RENDERING_H