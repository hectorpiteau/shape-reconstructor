#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "../utils/Utils.hpp"
#include "../utils/SceneSettings.hpp"
#include "../view/Lines.hpp"

class Camera
{

public:
    Camera(GLFWwindow *window, std::shared_ptr<SceneSettings> sceneSettings);
    ~Camera();

    void SetPosition(glm::vec3);
    void SetPosition(float x, float y, float z);
    
    glm::vec3 GetPosition();
    
    glm::vec3 GetRight();
    glm::vec3 GetRealUp();
    glm::vec3 GetUp();
    glm::vec3 GetForward();
    
    void ComputeMatricesFromInputs();

    glm::mat4 GetViewMatrix();
    glm::mat4 GetProjectionMatrix();

    const float* GetWireframe();

private:
    std::shared_ptr<SceneSettings> m_sceneSettings;
    GLFWwindow *m_window;

    glm::vec2 m_previousCursorPos;
    double yDeltaAngle;

    /** Camera position in world space coordinates. */
    glm::vec3 m_pos;
    /** The target position in world space that the camera is looking at. */
    glm::vec3 m_target;
    /** The camera's up vector. */
    glm::vec3 m_up;

    glm::mat4 m_viewMatrix;
    glm::mat4 m_projectionMatrix;

    glm::mat4 m_extrinsics; /** Rotate and translate compare to world coordinates. */
    glm::mat4 m_intrinsics; /** Projection of points from camera-space to image-space. */

    struct ScreenInfos m_screenInfos;

    float m_scroll = 0.0f;

    float m_speed = 3.0f;
    float m_horizontalAngle = 3.14f*1.25f;
    // Initial vertical angle : none
    float m_verticalAngle = -3.14f * 0.2f;
    // Initial Field of View
    float m_initialFoV = 65.0f;

    float m_mouseSpeed = 0.005f;

    float m_wireframeVertices[16*3] = {0.0f};
};