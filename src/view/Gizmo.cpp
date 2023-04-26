#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <memory>
#include "Gizmo.hpp"
#include "../controllers/Scene/Scene.hpp"
#include "../utils/Utils.hpp"

using namespace glm;

Gizmo::Gizmo(Scene* scene, const vec3& origin, const vec3& x, const vec3& y, const vec3& z) 
: SceneObject{std::string("Gizmo"), SceneObjectTypes::GIZMO}, m_scene(scene), m_origin(origin), m_x(x), m_y(y), m_z(z)
{
    WRITE_VEC3(m_xData, 0, m_origin);
    WRITE_VEC3(m_xData, 3, m_origin + m_x);

    WRITE_VEC3(m_yData, 0, m_origin);
    WRITE_VEC3(m_yData, 3, m_origin + m_y);
    
    WRITE_VEC3(m_zData, 0, m_origin);
    WRITE_VEC3(m_zData, 3, m_origin + m_z);

    m_xLines = std::unique_ptr<Lines>(new Lines(scene, m_xData, 6));
    m_yLines = std::unique_ptr<Lines>(new Lines(scene, m_yData, 6));
    m_zLines = std::unique_ptr<Lines>(new Lines(scene, m_zData, 6));

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

    // m_xLines = std::unique_ptr<Lines>(new Lines(m_scene, m_xData, 6));
    // m_yLines = std::unique_ptr<Lines>(new Lines(m_scene, m_yData, 6));
    // m_zLines = std::unique_ptr<Lines>(new Lines(m_scene, m_zData, 6));

    // m_xLines->SetColor(m_xColor);
    // m_yLines->SetColor(m_yColor);
    // m_zLines->SetColor(m_zColor);

    
    m_xLines->UpdateVertices(m_xData);
    m_yLines->UpdateVertices(m_yData);
    m_zLines->UpdateVertices(m_zData);
}

void Gizmo::SetPosition(const vec3& pos){
    m_origin = pos;
    // UpdateLines();
}

void Gizmo::SetX(const vec3& x){
    m_x = x;
    // UpdateLines();
}

void Gizmo::SetY(const vec3& y){
    m_y = y;
    // UpdateLines();
}

void Gizmo::SetZ(const vec3& z){
    m_z = z;
    // UpdateLines();
}

void Gizmo::Render()
{
    m_xLines->Render();
    m_yLines->Render();
    m_zLines->Render();
}   