#pragma once
#include <glm/glm.hpp>
#include "../../utils/SceneSettings.hpp"
#include <memory>

/**
 * @brief This interface is dedicated to be added to all objects that 
 * has in one way a wireframe representation. 
 * 
 * It allows to render the object's lines on screen.
 */
class Wireframe {
public:
    /**
     * @brief Updates a wireframe on screen given a projection and view matrix.
     * 
     * @param projection_intrinsics : The current camera's projection matrix.
     * @param view_extrinsics : The current camera's view matrix.
     * @param scene : A set of informations of the current scene / viewport.
     */
    virtual void UpdateWireFrame(const glm::mat4 &, const glm::mat4 &, std::shared_ptr<SceneSettings> scene) = 0;

    virtual ~Wireframe() {};
};