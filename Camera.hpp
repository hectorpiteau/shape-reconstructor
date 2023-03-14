#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "Utils.hpp"
#include "Lines.hpp"

class Camera
{

public:
    Camera(GLFWwindow *window, struct ScreenInfos screenInfos);
    ~Camera();

    void SetPosition(glm::vec3);
    void SetPosition(float x, float y, float z);
    glm::vec3 GetPosition();

    void OnKeyboard(unsigned char key);
    
    void ComputeMatricesFromInputs();

    glm::mat4 GetViewMatrix();
    glm::mat4 GetProjectionMatrix();

    void RenderWireframe();

private:
    GLFWwindow *m_window;
    glm::vec3 m_pos;
    glm::vec3 m_target;
    glm::vec3 m_up;

    glm::mat4 m_viewMatrix;
    glm::mat4 m_projectionMatrix;
    struct ScreenInfos m_screenInfos;

    float m_speed = 3.0f;
    float m_horizontalAngle = 3.14f*1.25f;
    // Initial vertical angle : none
    float m_verticalAngle = -3.14f * 0.2f;
    // Initial Field of View
    float m_initialFoV = 65.0f;

    float m_mouseSpeed = 0.005f;

    float m_wireframeVertices[16*3] = {0.0f};
    Lines m_lines;
};