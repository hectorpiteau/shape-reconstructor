#include <vector>
#include <iostream>
#include <memory>

#include "Camera.hpp"
#include "../ImageSet.hpp"
#include "CameraSet.hpp"
#include "../../controllers/Scene/Scene.hpp"

CameraSet::CameraSet()
    : m_areCameraGenerated(false), m_isLocked(false)
{
}

CameraSet::~CameraSet()
{
}

size_t CameraSet::Size()
{
    return m_cameras.size();
}

void CameraSet::AddCamera(std::shared_ptr<Camera> camera)
{
    m_cameras.push_back(camera);
}

std::shared_ptr<Camera> &CameraSet::operator[](size_t index)
{
    return m_cameras[index];
}

std::shared_ptr<Camera> CameraSet::GetCameraById(size_t id)
{
    for (std::shared_ptr<Camera> &cam : m_cameras)
    {
        if (cam->GetID() == id)
            return cam;
    }
    return nullptr;
}

std::vector<std::shared_ptr<Camera>> &CameraSet::GetCameras()
{
    return m_cameras;
}

bool CameraSet::AreCamerasGenerated()
{
    return m_areCameraGenerated;
}

bool CameraSet::IsLocked()
{
    return m_isLocked;
}

bool CameraSet::LinkToImageSet(std::shared_ptr<ImageSet> imageSet, std::shared_ptr<Scene> scene)
{
    if (imageSet == nullptr || scene == nullptr)
        return false;

    /** If there is already cameras in our list of cameras, empty it. */

    for (Image *img : imageSet->GetImages())
    {
        std::shared_ptr<Camera> cam = std::make_shared<Camera>(scene->GetWindow(), scene->GetSceneSettings());
        scene->Add(cam, true, true);

        m_cameras.push_back(cam);
    }

    m_isLocked = true;
    m_areCameraGenerated = true;
    m_areCalibrated = false;
}

bool CameraSet::CalibrateFromInformations(const std::vector<struct CameraCalibrationInformations> &informations)
{
    if (informations.size() != m_cameras.size())
        return false;

    std::cout << "CameraSet:: Calibrate cameras from informations. " << std::endl;

    for (int i = 0; i < informations.size(); ++i)
    {
        struct CameraCalibrationInformations info = informations[i];
        std::shared_ptr<Camera> cam = m_cameras[i];

        cam->SetIntrinsic(info.intrisic);
        cam->SetExtrinsic(info.extrinsic);
        cam->SetFovX(info.fov, true);
    }

    m_areCalibrated = true;
}

void CameraSet::Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene)
{
    for (auto &child : m_children)
    {
        child->Render(projection, view, scene);
    }
}