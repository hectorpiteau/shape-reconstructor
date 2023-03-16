#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include "../maths/MMath.hpp"
#include "../utils/Utils.hpp"
#include "Camera.hpp"
#include <memory>

Camera::Camera(GLFWwindow *window, struct ScreenInfos screenInfos)
{
    m_window = window;
    m_pos = glm::vec3(2.0f, 2.0f, 2.0f);
    m_target = glm::vec3(0.0f, 0.0f, 1.0f);
    m_up = glm::vec3(0.0f, 1.0f, 0.0f);
    m_screenInfos = screenInfos;
}

Camera::~Camera(){
    // delete m_lines;
}

glm::vec3 Camera::GetPosition()
{
    return m_pos;
}

void Camera::SetPosition(glm::vec3 pos)
{
    m_pos = pos;
}

void Camera::SetPosition(float x, float y, float z)
{
    m_pos.x = x;
    m_pos.y = y;
    m_pos.z = z;
}

void Camera::OnKeyboard(unsigned char Key)
{
}

glm::mat4 Camera::GetViewMatrix()
{
    return m_viewMatrix;
}

glm::mat4 Camera::GetProjectionMatrix()
{
    return m_projectionMatrix;
}

void Camera::ComputeMatricesFromInputs()
{
    static double lastTime = glfwGetTime();
    // Compute time difference between current and last frame
    double currentTime = glfwGetTime();
    float deltaTime = float(currentTime - lastTime);

    // Get mouse position
    double xpos, ypos;
    glfwGetCursorPos(m_window, &xpos, &ypos);

    // Reset mouse position for next frame
    glfwSetCursorPos(m_window, m_screenInfos.width / 2, m_screenInfos.height / 2);

    // Compute new orientation
    m_horizontalAngle += m_mouseSpeed * float(m_screenInfos.width / 2 - xpos);
    m_verticalAngle += m_mouseSpeed * float(m_screenInfos.height / 2 - ypos);

    // Direction : Spherical coordinates to Cartesian coordinates conversion
    glm::vec3 direction(
        cos(m_verticalAngle) * sin(m_horizontalAngle),
        sin(m_verticalAngle),
        cos(m_verticalAngle) * cos(m_horizontalAngle));

    // Right vector
    glm::vec3 right = glm::vec3(
        sin(m_horizontalAngle - 3.14f / 2.0f),
        0,
        cos(m_horizontalAngle - 3.14f / 2.0f));

    // Up vector
    glm::vec3 up = glm::cross(right, direction);

    // Move forward
    if (glfwGetKey(m_window, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS)
    {
        m_pos += direction * deltaTime * m_speed;
        // std::cout << "forward" << std::endl;
    }
    // Move backward
    if (glfwGetKey(m_window, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS)
    {
        m_pos -= direction * deltaTime * m_speed;
        // std::cout << "backward" << std::endl;
    }
    // Strafe right
    if (glfwGetKey(m_window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS)
    {
        m_pos += right * deltaTime * m_speed;
        // std::cout << "right" << std::endl;
    }
    // Strafe left
    if (glfwGetKey(m_window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS)
    {
        m_pos -= right * deltaTime * m_speed;
        // std::cout << "left" << std::endl;
    }

    float FoV = m_initialFoV; // - 5 * glfwGetMouseWheel(); // Now GLFW 3 requires setting up a callback for this. It's a bit too complicated for this beginner's tutorial, so it's disabled instead.

    // Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    m_projectionMatrix = glm::perspective(glm::radians(FoV), 4.0f / 3.0f, 0.1f, 100.0f);
    // Camera matrix
    m_viewMatrix = glm::lookAt(
        m_pos,             // Camera is here
        m_pos + direction, // and looks here : at the same position, plus "direction"
        m_up               // Head is up (set to 0,-1,0 to look upside-down)
    );

    // For the next frame, the "last time" will be "now"
    lastTime = currentTime;

    // glm::mat4 CameraTransformation = MMath::InitCameraTransform(m_pos, _target, _up);
    // return CameraTransformation;
}

void Camera::RenderWireframe()
{
    glm::vec3 forward(
        cos(m_verticalAngle) * sin(m_horizontalAngle),
        sin(m_verticalAngle),
        cos(m_verticalAngle) * cos(m_horizontalAngle)
    );

    glm::vec3 right = glm::vec3(sin(m_horizontalAngle - 3.14f / 2.0f), 0, cos(m_horizontalAngle - 3.14f / 2.0f));

    float length = 1.2f;

    glm::vec3 camera_origin = m_pos;

    glm::vec3 corner_top_left = m_pos + forward * length + m_up - right;
    glm::vec3 corner_top_right = m_pos + forward * length + m_up + right;

    glm::vec3 corner_bot_left = m_pos + forward * length - m_up - right;
    glm::vec3 corner_bot_right = m_pos + forward * length - m_up + right;

    // m_wireframeVertices[0] = corner_top_left;
    // m_wireframeVertices[1] = corner_top_right;

    // m_wireframeVertices[2] = corner_bot_left;
    // m_wireframeVertices[3] = corner_bot_right;

    // m_wireframeVertices[4] = corner_top_left;
    // m_wireframeVertices[5] = corner_bot_left;

    // m_wireframeVertices[6] = corner_top_right;
    // m_wireframeVertices[7] = corner_bot_right;

    // m_wireframeVertices[8] = camera_origin;
    // m_wireframeVertices[9] = corner_top_left;

    // m_wireframeVertices[10] = camera_origin;
    // m_wireframeVertices[11] = corner_top_right;

    // m_wireframeVertices[12] = camera_origin;
    // m_wireframeVertices[13] = corner_bot_left;

    // m_wireframeVertices[14] = camera_origin;
    // m_wireframeVertices[15] = corner_bot_right;
}