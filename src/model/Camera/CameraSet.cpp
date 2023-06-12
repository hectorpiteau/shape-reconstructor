#include <vector>
#include <iostream>
#include <memory>

#include "Camera.hpp"
#include "../ImageSet.hpp"
#include "CameraSet.hpp"
#include "../../controllers/Scene/Scene.hpp"

#include "../../include/icons/IconsFontAwesome6.h"

CameraSet::CameraSet(Scene *scene)
    : SceneObject{std::string("CameraSet"), SceneObjectTypes::CAMERASET}, 
    m_areCameraGenerated(false),
    m_areCalibrated(false),
    m_isLocked(false),
    m_scene(scene)
{
    SetName(std::string("Camera Set"));
}

CameraSet::~CameraSet() = default;

size_t CameraSet::Size()
{
    return m_cameras.size();
}

void CameraSet::AddCamera(std::shared_ptr<Camera> camera)
{
    m_cameras.push_back(camera);
    m_children.push_back(camera);
}

std::shared_ptr<Camera> &CameraSet::operator[](size_t index)
{
    return m_cameras[index];
}

std::shared_ptr<Camera> CameraSet::GetCameraById(int id)
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

bool CameraSet::AreCamerasGenerated() const
{
    return m_areCameraGenerated;
}

bool CameraSet::IsLocked() const
{
    return m_isLocked;
}

bool CameraSet::LinkToImageSet(std::shared_ptr<ImageSet> imageSet)
{
    if (imageSet == nullptr || m_scene == nullptr)
    {
        std::cout << "CameraSet::LinkToImageSet : imageset null or scene null. " << std::endl;
        return false;
    }

    /** If there is already cameras in our list of cameras, empty it. */
    int cpt = 0;
    for (Image *img : imageSet->GetImages())
    {
        std::shared_ptr<Camera> cam = std::make_shared<Camera>(m_scene);
        cam->SetActive(true);
        cam->SetIsChild(true);
        cam->SetName(std::string(ICON_FA_CAMERA " Camera ") + std::to_string(cpt++));
        cam->SetIsVisibleInList(false);
        cam->SetImage(img);
        m_scene->Add(cam, true, true);
        std::cout << "CameraSet: Add camera for img: " << img->filename << " " << cam->GetID() << std::endl;

        m_cameras.push_back(cam);
        m_children.push_back(cam);
    }

    m_isLocked = true;
    m_areCameraGenerated = true;
    m_areCalibrated = false;
    return true;
}

bool CameraSet::CalibrateFromInformation(const std::vector<CameraCalibrationInformations> &information)
{
    if (information.size() != m_cameras.size())
        return false;

    for (size_t i = 0; i < information.size(); ++i)
    {
        CameraCalibrationInformations info = information[i];
        std::shared_ptr<Camera> cam = m_cameras[i];

        std::cout << "Calibrate camera: " << i << std::endl;

        cam->SetIntrinsic(info.intrinsic);
        cam->SetExtrinsic(info.extrinsic);
    }

    m_areCalibrated = true;
    return true;
}

void CameraSet::Render()
{
    for (auto &cam : m_cameras)
        cam->Render();
}

void CameraSet::ShowCenterLines()
{
    for (auto &cam : m_cameras)
        cam->SetIsCenterLineVisible(true);
}

void CameraSet::HideCenterLines()
{
    for (auto &cam : m_cameras)
        cam->SetIsCenterLineVisible(false);
}

void CameraSet::SetCenterLinesLength(float length)
{
    for (auto &cam : m_cameras)
        cam->SetCenterLineLength(length);
}

float CameraSet::GetFrustumSize() const {
    return m_frustumSize;
}

void CameraSet::SetFrustumSize(float value){
    m_frustumSize = value;
    for (auto &cam : m_cameras)
        cam->SetFrustumSize(value);
}
