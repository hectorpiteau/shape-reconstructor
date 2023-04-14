#include <memory>
#include <glm/glm.hpp>

#include "../model/Camera/Camera.hpp"
#include "CameraInteractor.hpp"

CameraInteractor::CameraInteractor()
{
}

void CameraInteractor::SetCamera(std::shared_ptr<Camera> camera)
{
    m_camera = camera;
}

void CameraInteractor::SetIsActive(bool active)
{
}

void CameraInteractor::SetPosition(const glm::vec3 &pos)
{
    m_camera->SetPosition(pos);
}

void CameraInteractor::SetTarget(const glm::vec3 &target)
{
    return m_camera->SetTarget(target);
}

void CameraInteractor::SetUp(const glm::vec3 &up)
{
    // m_camera->SetUp(up);
}

void CameraInteractor::SetFovX(float fov)
{
    m_camera->SetFovX(fov);
}

void CameraInteractor::SetFovY(float fov)
{
    m_camera->SetFovX(fov);
}

void CameraInteractor::SetNear(float near)
{
    m_camera->SetNear(near);
}

float CameraInteractor::GetNear()
{
    return m_camera->GetNear();
}

void CameraInteractor::SetFar(float far)
{
    m_camera->SetFar(far);
}

float CameraInteractor::GetFar()
{
    return m_camera->GetFar();
}

void CameraInteractor::SetDistortion(const glm::vec2 &dist)
{
    // m_camera->SetDistortion(dist);
}


const glm::vec2 CameraInteractor::GetDistortion()
{
    return glm::vec2(0.0f);
}

bool CameraInteractor::GetIsActive()
{
    return m_camera->IsActive();
}

const glm::vec3 &CameraInteractor::GetPosition()
{
    return m_camera->GetPosition();
}

const glm::vec3 &CameraInteractor::GetTarget()
{
    // return m_camera->
}

const glm::vec3& CameraInteractor::GetUp()
{
    return m_camera->GetUp();
}

float CameraInteractor::GetFovX()
{
    return m_camera->GetFovX();
}

float CameraInteractor::GetFovY()
{
    return m_camera->GetFovY();
}

std::shared_ptr<Camera>& CameraInteractor::GetCamera(){
    return m_camera;
}