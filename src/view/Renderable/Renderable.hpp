#pragma once
#include <glm/glm.hpp>
#include "../../utils/SceneSettings.hpp"
#include <memory>

class Renderable {
public:
    virtual void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene) = 0;
    virtual ~Renderable() {};
};