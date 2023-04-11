#pragma once

#include <memory>
#include <glm/glm.hpp>

#include "../model/Camera/Camera.hpp"

/**
 * @brief Camera interactor.
 * Used for the view to interact and edit a camera.
 */
class CameraInteractor {
public:
    /**
     * @brief Construct a new Camera Interactor object.
     */
    CameraInteractor();

    /**
     * @brief Set the Camera to be edited.
     * 
     * @param camera : A smart pointer of the active camera to edit.
     */
    void SetCamera(std::shared_ptr<Camera> camera);

    /**
     * @brief Set the Is Active object
     * 
     * @param active 
     */
    void SetIsActive(bool active);
    /**
     * @brief Get the Is Active object
     * 
     * @return true 
     * @return false 
     */
    bool GetIsActive();

    void SetPosition(const glm::vec3& pos);
    const glm::vec3& GetPosition();

    void SetTarget(const glm::vec3& target);
    const glm::vec3& GetTarget();

    void SetUp(const glm::vec3& up);
    const glm::vec3& GetUp();

    void SetFovX(float fov);
    float GetFovX();
    
    void SetFovY(float fov);
    float GetFovY();
    
    void SetNear(float near);
    float GetNear();

    void SetFar(float far);
    float GetFar();

    void SetDistortion(const glm::vec2& dist);
    const glm::vec2& GetDistortion(); 

    std::shared_ptr<Camera>& GetCamera();

private:
    /** A pointer to the camera to modify. */
    std::shared_ptr<Camera> m_camera;
};