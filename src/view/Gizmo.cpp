#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <memory>
#include "Gizmo.hpp"

#include "../utils/Utils.hpp"

Gizmo::Gizmo(glm::vec3 origin, glm::vec3 x, glm::vec3 y, glm::vec3 z) 
: m_xVec(x), m_yVec(y), m_zVec(z)
{
    WRITE_VEC3(m_xData, 0, origin);
    WRITE_VEC3(m_xData, 3, origin + x);

    WRITE_VEC3(m_yData, 0, origin);
    WRITE_VEC3(m_yData, 3, origin + y);
    
    WRITE_VEC3(m_zData, 0, origin);
    WRITE_VEC3(m_zData, 3, origin+ z);

    m_xLines = std::unique_ptr<Lines>(new Lines(m_xData, 6));
    m_yLines = std::unique_ptr<Lines>(new Lines(m_yData, 6));
    m_zLines = std::unique_ptr<Lines>(new Lines(m_zData, 6));

    m_xLines->SetColor(m_xColor);
    m_yLines->SetColor(m_yColor);
    m_zLines->SetColor(m_zColor);
}

void Gizmo::Render(glm::mat4 &projectionMatrix, glm::mat4 &viewMatrix)
{
    m_xLines->Render(projectionMatrix, viewMatrix);
    m_yLines->Render(projectionMatrix, viewMatrix);
    m_zLines->Render(projectionMatrix, viewMatrix);
}