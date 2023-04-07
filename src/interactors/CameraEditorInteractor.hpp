#pragma once
#include <memory>
#include "../model/Camera/MCamera.hpp"
#include <glm/glm.hpp>

class CameraEditorInteractor {
public:
    CameraEditorInteractor(std::shared_ptr<MCamera> camera) : m_camera(camera){

    }

    void SetCamera(std::shared_ptr<MCamera> camera);

    void SetAsActive();

    void SetPosition(const glm::vec3& pos);
    
    void SetTarget(const glm::vec3& target);

    void SetUp(const glm::vec3& up);

    void SetFov(float fov);
    
    void SetNear(float near);

    void SetFar(float far);

    void SetDistortion(const glm::vec2& dist);

private:
    std::shared_ptr<MCamera> m_camera;
};