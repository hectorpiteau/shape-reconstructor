#pragma once

#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <memory>
#include "Lines.hpp"
#include "SceneObject/SceneObject.hpp"
#include "../controllers/Scene/Scene.hpp"

class Lines;
class Scene;

using namespace glm;

class Gizmo : public SceneObject {
public:
    Gizmo(Scene* scene, const vec3& origin, const vec3& x, const vec3& y, const vec3& z);

    void UpdateLines();

    void SetPosition(const vec3& pos);
    void SetX(const vec3& x);
    void SetY(const vec3& y);
    void SetZ(const vec3& z);
    
    void Render();
    
private:
    Scene* m_scene;
    float m_xLength = 1.0f;
    float m_yLength = 1.0f;
    float m_zLength = 1.0f;

    float m_xData[6];
    std::vector<vec3> m_xVec;
    float m_yData[6];
    std::vector<vec3> m_yVec;
    float m_zData[6];
    std::vector<vec3> m_zVec;

    vec3 m_origin = vec3(0.0, 0.0, 0.0);
    vec3 m_x = vec3(1.0, 0.0, 0.0);
    vec3 m_y = vec3(0.0, 1.0, 0.0);
    vec3 m_z = vec3(0.0, 0.0, 1.0);

    std::unique_ptr<Lines> m_xLines;
    std::unique_ptr<Lines> m_yLines;
    std::unique_ptr<Lines> m_zLines;

    vec4 m_xColor = vec4(1.0, 0.0, 0.0, 1.0);
    vec4 m_yColor = vec4(0.0, 1.0, 0.0, 1.0);
    vec4 m_zColor = vec4(0.0, 0.0, 1.0, 1.0);
};
