#include <memory>
#include <iostream>

#include "Volume3D.hpp"

#include "Lines.hpp"
#include "Renderable/Renderable.hpp"
#include "Wireframe/Wireframe.hpp"
#include "SceneObject/SceneObject.hpp"

#include "../../include/icons/IconsFontAwesome6.h"

Volume3D::Volume3D() : SceneObject{std::string("VOLUME3D")}
{
    SetName(std::string(ICON_FA_CUBES " Volume 3D"));
    m_lines = std::make_shared<Lines>(m_wireframeVertices, 12 * 2 * 3);
}

void Volume3D::Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene)
{
    /** nothing special here for now */
}

void Volume3D::UpdateWireFrame(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene)
{
    m_lines->Render(projection, view, scene);
}
