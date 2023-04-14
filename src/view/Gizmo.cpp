#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <memory>
#include "Gizmo.hpp"

#include "../utils/Utils.hpp"

Gizmo::Gizmo(const glm::vec3& origin, const glm::vec3& x, const glm::vec3& y, const glm::vec3& z) 
: m_origin(origin), m_x(x), m_y(y), m_z(z)
{
    UpdateLines();

    m_xLines = std::unique_ptr<Lines>(new Lines(m_xData, 6));
    m_yLines = std::unique_ptr<Lines>(new Lines(m_yData, 6));
    m_zLines = std::unique_ptr<Lines>(new Lines(m_zData, 6));

    m_xLines->SetColor(m_xColor);
    m_yLines->SetColor(m_yColor);
    m_zLines->SetColor(m_zColor);
}

void Gizmo::UpdateLines(){
    WRITE_VEC3(m_xData, 0, m_origin);
    WRITE_VEC3(m_xData, 3, m_origin + m_x);

    WRITE_VEC3(m_yData, 0, m_origin);
    WRITE_VEC3(m_yData, 3, m_origin + m_y);
    
    WRITE_VEC3(m_zData, 0, m_origin);
    WRITE_VEC3(m_zData, 3, m_origin + m_z);
}

void Gizmo::SetPosition(const glm::vec3& pos){
    m_origin = pos;
    UpdateLines();
}

void Gizmo::SetX(const glm::vec3& x){
    m_x = x;
    UpdateLines();
}

void Gizmo::SetY(const glm::vec3& y){
    m_y = y;
    UpdateLines();
}

void Gizmo::SetZ(const glm::vec3& z){
    m_z = z;
    UpdateLines();
}

void Gizmo::Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene)
{
    m_xLines->Render(projection, view, scene);
    m_yLines->Render(projection, view, scene);
    m_zLines->Render(projection, view, scene);
}   