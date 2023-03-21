#pragma once
#include <glm/glm.hpp>
#include "../../utils/SceneSettings.hpp"
#include <memory>

/**
 * @brief This interface is meant to be inherited by any object willing to be rendered on screen.
 * 
 * It provides a generic method called Render.
 */
class Renderable {
public:
    /**
     * @brief Render the object on screen given a specific camera settings.
     * 
     * @param projection : The current camera's projection matrix.
     * @param view : The current camera's view matrix.
     * @param scene : A set of informations on the current scene / viewport.
     */
    virtual void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene) = 0;
    virtual ~Renderable() {};
};