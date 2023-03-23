#pragma once
#include "../utils/SceneSettings.hpp"
#include <glm/glm.hpp>
#include <memory>

class RayMarcher
{
public:
    RayMarcher(std::shared_ptr<SceneSettings> sceneSettings) : m_sceneSettings(sceneSettings){};

    glm::vec3 GetRay()
    {
        
    }

private:
    std::shared_ptr<SceneSettings> m_sceneSettings;
};