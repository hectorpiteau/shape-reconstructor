#pragma once

#include <glm/glm.hpp>

/**
 * @brief Most simple representation of a camera.
 * 
 * Include extrinsic and intrinsic parameters.
 * Include distortion parameters.
 */
class MCamera {
public:
    MCamera();
    MCamera(const MCamera&) = delete;
    ~MCamera();

    const glm::mat& GetExtrinsic();
    void SetExtrinsic(const glm::mat&);
    
    const glm::mat& GetIntrinsic();
    void SetIntrinsic(const glm::mat&);

private:

    glm::mat4 m_extrinsic;
    glm::mat4 m_intrinsic;

    glm::vec2 m_dist;

};