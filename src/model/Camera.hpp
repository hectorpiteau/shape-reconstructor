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

    /**
     * @brief Set the camera's position in world space coordinates.
     * 
     * @param position : The camera position in world space coordinates.
     */
    void SetPosition(const glm::vec3& position);

    /**
     * @brief Set the camera's position in world space coordinates.
     * 
     * @param x : The x coordinate in world space coordinates.
     * @param y : The y coordinate in world space coordinates.
     * @param z : The z coordinate in world space coordinates.
     */
    void SetPosition(float x, float y, float z);
    
    /**
     * @brief Get the camera's position in world space.
     * 
     * @return glm::vec3 : World position.
     */
    glm::vec3 GetPosition();
    
    /**
     * @brief Get the camera's right vector, in world space coordinates.
     * 
     * @return glm::vec3 : Right vector.
     */
    glm::vec3 GetRight();

    /**
     * @brief Get the camera's real up vector, (the one orthogonal to 
     * forward and right vectors), in world space coordinates. 
     * 
     * @return glm::vec3 : Real up vector.
     */
    glm::vec3 GetRealUp();

    /**
     * @brief Get the camera's up vector in world space coordinates.
     * 
     * @return glm::vec3 : Up vector.
     */
    glm::vec3 GetUp();

    /**
     * @brief Get the camera's forward vector in world space coordinates.
     * 
     * @return glm::vec3 : Forward vector.
     */
    glm::vec3 GetForward();

    /**
     * @brief Get the camera's target (lookAt) in world space coordinates.
     * Where the camera is looking at.
     * 
     * @return glm::vec3 : The camera lookAt point in world space coordinates.
     */
    glm::vec3 GetTarget();
    
    /**
     * @brief Computes the View / Projection (extrinsic, intrinsic) matrices 
     * from the mouse and keyboard inputs.
     */
    void ComputeMatricesFromInputs(GLFWwindow *window);

    /**
     * @brief Get the View Matrix also known as the extrinsics matrix.
     * 
     * @return const glm::mat4& : A constant reference to the matrix.
     */
    const glm::mat4& GetViewMatrix();

    /**
     * @brief Get the Projection Matrix also known as the intrinsics matrix.
     * 
     * @return const glm::mat4& 
     */
    const glm::mat4& GetProjectionMatrix();

    /**
     * @brief Get the camera's Wireframe representation.
     * //TODO: Move in the view. (Detach dependencie). 
     * 
     * @return const float* : A constant list of floats that represents points in space that defines
     * a list of lines (wireframe).
     */
    const float* GetWireframe();

private:
    std::shared_ptr<SceneSettings> m_sceneSettings;

    glm::vec2 m_previousCursorPos;
    double yDeltaAngle;

    /** Camera position in world space coordinates. */
    glm::vec3 m_pos;
    /** The target position in world space that the camera is looking at. */
    glm::vec3 m_target;
    /** The camera's up vector. */
    glm::vec3 m_up;

    /**
     * @brief Also known as the extrinsic matrix.
     * Rotate and translate compare to world coordinates.
     */
    glm::mat4 m_viewMatrix;

    /**
     * @brief Also known as the intrinsic matrix.
     * Projection of points from camera-space to image-space.
     */
    glm::mat4 m_projectionMatrix;

    float m_scroll = 0.0f;
    float m_speed = 3.0f;
    float m_horizontalAngle = 3.14f*1.25f;
    // Initial vertical angle : none
    float m_verticalAngle = -3.14f * 0.2f;
    // Initial Field of View
    float m_initialFoV = 90.0f;

    float m_mouseSpeed = 0.005f;

    float m_wireframeVertices[16*3] = {0.0f};
};