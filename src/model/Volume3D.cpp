#include <memory>
#include <iostream>

#include "Volume3D.hpp"

#include "../view/Lines.hpp"
#include "../view/Renderable/Renderable.hpp"
#include "../view/Wireframe/Wireframe.hpp"
#include "../view/SceneObject/SceneObject.hpp"

#include "../controllers/Scene/Scene.hpp"

#include "../../include/icons/IconsFontAwesome6.h"

Volume3D::Volume3D(Scene* scene) : SceneObject{std::string("VOLUME3D"), SceneObjectTypes::VOLUME3D}
{
    SetName(std::string(ICON_FA_CUBES " Volume 3D"));
    m_lines = std::make_shared<Lines>(scene, m_wireframeVertices, 12 * 2 * 3);
}

void Volume3D::Render()
{
    /** nothing special here for now */
    m_lines->Render();
}

void Volume3D::UpdateWireFrame(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene)
{
    
}
