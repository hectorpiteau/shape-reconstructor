#pragma once

#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <memory>
#include "Lines.hpp"
#include "Renderable/Renderable.hpp"

class Gizmo : Renderable {
public:
    Gizmo(glm::vec3 origin, glm::vec3 x, glm::vec3 y, glm::vec3 z);

    void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene);
    
private:
    float m_xLength;
    float m_yLength;
    float m_zLength;

    float m_xData[6];
    float m_yData[6];
    float m_zData[6];

    glm::vec3 m_xVec = glm::vec3(1.0, 0.0, 0.0);
    glm::vec3 m_yVec = glm::vec3(0.0, 1.0, 0.0);
    glm::vec3 m_zVec = glm::vec3(0.0, 0.0, 1.0);

    std::unique_ptr<Lines> m_xLines;
    std::unique_ptr<Lines> m_yLines;
    std::unique_ptr<Lines> m_zLines;

    glm::vec4 m_xColor = glm::vec4(1.0, 0.0, 0.0, 1.0);
    glm::vec4 m_yColor = glm::vec4(0.0, 1.0, 0.0, 1.0);
    glm::vec4 m_zColor = glm::vec4(0.0, 0.0, 1.0, 1.0);

    
};
