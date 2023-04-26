#include <string>
#include <memory>
#include <vector>

#include "../model/Camera/CameraSet.hpp"
#include "../model/Camera/Camera.hpp"
#include "../model/ImageSet.hpp"

#include "../view/SceneObject/SceneObject.hpp"
#include "../controllers/Scene/Scene.hpp"

#include "CameraSetInteractor.hpp"

CameraSetInteractor::CameraSetInteractor(Scene* scene) : m_scene(scene), m_dummyCameras() {}
CameraSetInteractor::~CameraSetInteractor() {}

void CameraSetInteractor::SetActiveCameraSet(std::shared_ptr<CameraSet> cameraSet) {
    m_cameraSet = cameraSet;
    m_centerLinesLength = 1.0f;
}

std::vector<std::shared_ptr<Camera>>& CameraSetInteractor::GetCameras() {
    return m_cameraSet == nullptr ? m_dummyCameras : m_cameraSet->GetCameras();
}

size_t CameraSetInteractor::GetAmountOfCameras() {
    return m_cameraSet == nullptr ? 0 : m_cameraSet->Size();
}

bool CameraSetInteractor::AreCamerasGenerated() {
    return m_cameraSet == nullptr ? false : m_cameraSet->AreCamerasGenerated();
}

bool CameraSetInteractor::IsCameraSetLocked() {
    return m_cameraSet == nullptr ? false : m_cameraSet->IsLocked();
}

bool CameraSetInteractor::LinkCameraSetToSceneObject(int id) {
    if(m_cameraSet == nullptr) return false;
    std::shared_ptr<SceneObject> obj = m_scene->Get(id);
    
    if(obj == nullptr){
        return false;
    }else{
        if(obj->GetType() == SceneObjectTypes::IMAGESET){
            m_cameraSet->LinkToImageSet(std::dynamic_pointer_cast<ImageSet>(obj));
            return true;
        }
        return false;
    }
}


void CameraSetInteractor::ShowCenterLines(){
    m_cameraSet->ShowCenterLines();
}

void CameraSetInteractor::HideCenterLines(){
    m_cameraSet->HideCenterLines();
}

float CameraSetInteractor::GetCenterLinesLength(){
    return m_centerLinesLength;
}
void CameraSetInteractor::SetCenterLinesLength(float length){
    m_centerLinesLength = length;
    m_cameraSet->SetCenterLinesLength(length);
}