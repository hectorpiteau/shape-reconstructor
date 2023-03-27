#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vector>
#include "../maths/MMath.hpp"
#include "../utils/Utils.hpp"
#include "../utils/SceneSettings.hpp"
#include "../utils/Projection.hpp"
#include "Camera.hpp"
#include <memory>

Camera::Camera(GLFWwindow *window, std::shared_ptr<SceneSettings> sceneSettings)
    : m_sceneSettings(sceneSettings)
{
    window = window;
    m_pos = glm::vec3(4.0f, 4.0f, 4.0f);
    m_target = glm::vec3(0.0f, 0.0f, 0.0f);
    m_up = glm::vec3(0.0f, 1.0f, 0.0f);
    m_projectionMatrix = glm::perspective(
        glm::radians(m_initialFoV), 
        m_sceneSettings->GetViewportRatio(), 
        0.01f, 
        100.0f
    );

    m_viewMatrix = glm::lookAt( m_pos, m_target, m_up);
    
    m_previousCursorPos = glm::vec2(m_sceneSettings->GetViewportWidth() / 2, m_sceneSettings->GetViewportHeight() / 2);
}

Camera::~Camera()
{
    // delete m_lines;
}

glm::vec3 Camera::GetPosition()
{
    return m_pos;
}

void Camera::SetPosition(const glm::vec3 &position)
{
    m_pos = position;
}

void Camera::SetPosition(float x, float y, float z)
{
    m_pos.x = x;
    m_pos.y = y;
    m_pos.z = z;
}

const glm::mat4& Camera::GetViewMatrix()
{
    return m_viewMatrix;
}

const glm::mat4& Camera::GetProjectionMatrix()
{
    return m_projectionMatrix;
}

void Camera::ComputeMatricesFromInputs(GLFWwindow *window)
{

    static double lastTime = glfwGetTime();

    /** Compute time difference between current and last frame */
    double currentTime = glfwGetTime();

    float deltaTime = float(currentTime - lastTime);

    // Get mouse position
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Reset mouse position for next frame
    if (m_sceneSettings->GetCameraModel() == CameraMovementModel::FPS)
    {
        glfwSetCursorPos(window, m_sceneSettings->GetViewportWidth() / 2, m_sceneSettings->GetViewportHeight() / 2);
        // Compute new orientation
        m_horizontalAngle += m_mouseSpeed * float(m_sceneSettings->GetViewportWidth() / 2 - xpos);
        m_verticalAngle += m_mouseSpeed * float(m_sceneSettings->GetViewportHeight() / 2 - ypos);
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
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        {
            m_pos += direction * deltaTime * m_speed;
        }
        // Move backward
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        {
            m_pos -= direction * deltaTime * m_speed;
        }
        // Strafe right
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        {
            m_pos += right * deltaTime * m_speed;
        }
        // Strafe left
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        {
            m_pos -= right * deltaTime * m_speed;
        }

        // Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
        // (intrinsics)
        // m_projectionMatrix = glm::perspective(glm::radians(m_initialFoV), m_sceneSettings->GetViewportRatio(), 0.1f, 100.0f);

        // Camera matrix
        // (extrinsics)
        m_viewMatrix = glm::lookAt(
            m_pos,             // Camera is here
            m_pos + direction, // and looks here : at the same position, plus "direction"
            m_up               // Head is up (set to 0,-1,0 to look upside-down)
        );
    }
    else
    {
        /** If the user is not left clicking, nothing happens in arcball mode. */
        if (m_sceneSettings->GetMouseLeftClick() == false)
        {
            m_previousCursorPos.x = xpos;
            m_previousCursorPos.y = ypos;
            lastTime = currentTime;
            return;
        }

        glm::vec3 viewDir = -glm::transpose(m_viewMatrix)[2];

        /** If the shift key is pressed, move is enabled and replaces the rotation. */
        if (m_sceneSettings->GetShiftKey())
        {

            float deltaX = 4.0f / m_sceneSettings->GetViewportWidth();
            float deltaY = 4.0f / m_sceneSettings->GetViewportHeight();

            float xDisplacement = (m_previousCursorPos.x - xpos) * deltaX;
            float yDisplacement = (m_previousCursorPos.y - ypos) * deltaY;

            glm::vec3 right = glm::transpose(m_viewMatrix)[0];
            glm::vec3 up = m_up;

            m_pos = m_pos + xDisplacement * right - up * yDisplacement;
            m_target = m_target + xDisplacement * right - up * yDisplacement;

            m_viewMatrix = glm::lookAt(
                m_pos + viewDir * m_sceneSettings->GetScrollOffsets().y,
                m_target + viewDir * m_sceneSettings->GetScrollOffsets().y,
                m_up);

            m_previousCursorPos.x = xpos;
            m_previousCursorPos.y = ypos;

            lastTime = currentTime;
            return;
        }

        // Get the homogenous position of the camera and pivot point
        glm::vec4 position(m_pos.x, m_pos.y, m_pos.z, 1);
        glm::vec4 pivot(m_target.x, m_target.y, m_target.z, 1);

        // step 1 : Calculate the amount of rotation given the mouse movement.
        float deltaAngleX = (2 * M_PI / m_sceneSettings->GetViewportWidth()); // a movement from left to right = 2*PI = 360 deg
        float deltaAngleY = (M_PI / m_sceneSettings->GetViewportHeight());    // a movement from top to bottom = PI = 180 deg
        float xAngle = (m_previousCursorPos.x - xpos) * deltaAngleX;
        float yAngle = (m_previousCursorPos.y - ypos) * deltaAngleY;

        // Extra step to handle the problem when the camera direction is the same as the up vector
        float cosAngle = glm::dot(viewDir, m_up);
        if (cosAngle * Utils::Sign(yDeltaAngle) > 0.99f)
            yDeltaAngle = 0;

        // step 2: Rotate the camera around the pivot point on the first axis.
        glm::mat4x4 rotationMatrixX(1.0f);
        rotationMatrixX = glm::rotate(rotationMatrixX, xAngle, m_up);
        position = (rotationMatrixX * (position - pivot)) + pivot;

        // step 3: Rotate the camera around the pivot point on the second axis.
        glm::mat4x4 rotationMatrixY(1.0f);
        glm::vec3 right = glm::transpose(m_viewMatrix)[0];
        rotationMatrixY = glm::rotate(rotationMatrixY, yAngle, right);
        glm::vec3 finalPosition = (rotationMatrixY * (position - pivot)) + pivot;

        // Update the camera view (we keep the same lookat and the same up vector)
        m_pos = finalPosition;

        // + viewDir * m_sceneSettings->GetScrollOffsets().y
        m_viewMatrix = glm::lookAt(
            m_pos ,
            m_target,
            m_up);
    }

    // Update the mouse position for the next rotation
    m_previousCursorPos.x = xpos;
    m_previousCursorPos.y = ypos;

    // For the next frame, the "last time" will be "now"
    lastTime = currentTime;
}

glm::vec3 Camera::GetTarget(){
    return m_target;
}

glm::vec3 Camera::GetRight()
{
    // return glm::vec3(sin(m_horizontalAngle - 3.14f / 2.0f), 0, cos(m_horizontalAngle - 3.14f / 2.0f));
    return glm::normalize(glm::cross(GetForward(), GetUp()));
}

glm::vec3 Camera::GetRealUp(){
    return glm::normalize(glm::cross(GetRight(), GetForward()));
}
glm::vec3 Camera::GetUp()
{
    return m_up;
}

glm::vec3 Camera::GetForward()
{
    // return glm::vec3(
    //     cos(m_verticalAngle) * sin(m_horizontalAngle),
    //     sin(m_verticalAngle),
    //     cos(m_verticalAngle) * cos(m_horizontalAngle));

    return glm::normalize(m_pos - m_target);
}

const float *Camera::GetWireframe()
{
    glm::vec3 corner_top_left_tmp = Projection::NDCToCamera(glm::vec2(-1.0, 1.0), m_projectionMatrix);
    glm::vec3 corner_top_left = Projection::CameraToWorld(glm::vec4(corner_top_left_tmp, 1.0f), m_viewMatrix);
    
    glm::vec3 corner_top_right_tmp = Projection::NDCToCamera(glm::vec2(1.0, 1.0), m_projectionMatrix);
    glm::vec3 corner_top_right = Projection::CameraToWorld(glm::vec4(corner_top_right_tmp, 1.0f), m_viewMatrix);
    
    glm::vec3 corner_bot_left_tmp = Projection::NDCToCamera(glm::vec2(-1.0, -1.0), m_projectionMatrix);
    glm::vec3 corner_bot_left = Projection::CameraToWorld(glm::vec4(corner_bot_left_tmp, 1.0f), m_viewMatrix);
    
    glm::vec3 corner_bot_right_tmp = Projection::NDCToCamera(glm::vec2(1.0, -1.0), m_projectionMatrix);
    glm::vec3 corner_bot_right = Projection::CameraToWorld(glm::vec4(corner_bot_right_tmp, 1.0f), m_viewMatrix);
    

    WRITE_VEC3(m_wireframeVertices, 0, corner_top_left);
    WRITE_VEC3(m_wireframeVertices, 3, corner_top_right);

    WRITE_VEC3(m_wireframeVertices, 6, corner_bot_left);
    WRITE_VEC3(m_wireframeVertices, 9, corner_bot_right);

    WRITE_VEC3(m_wireframeVertices, 12, corner_top_left);
    WRITE_VEC3(m_wireframeVertices, 15, corner_bot_left);

    WRITE_VEC3(m_wireframeVertices, 18, corner_top_right);
    WRITE_VEC3(m_wireframeVertices, 21, corner_bot_right);

    WRITE_VEC3(m_wireframeVertices, 24, m_pos);
    WRITE_VEC3(m_wireframeVertices, 27, corner_top_left);

    WRITE_VEC3(m_wireframeVertices, 30, m_pos);
    WRITE_VEC3(m_wireframeVertices, 33, corner_top_right);

    WRITE_VEC3(m_wireframeVertices, 36, m_pos);
    WRITE_VEC3(m_wireframeVertices, 39, corner_bot_left);

    WRITE_VEC3(m_wireframeVertices, 42, m_pos);
    WRITE_VEC3(m_wireframeVertices, 45, corner_bot_right);

    return m_wireframeVertices;
}