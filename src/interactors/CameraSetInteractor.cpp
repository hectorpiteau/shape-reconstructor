#pragma once
#include <string>
#include <memory>

#include "../model/Camera/CameraSet.hpp"
#include "../model/Camera/Camera.hpp"
#include "../model/ImageSet.hpp"

#include "../view/SceneObject/SceneObject.hpp"
#include "../controllers/Scene/Scene.hpp"

#include "CameraSetInteractor.hpp"

CameraSetInteractor::CameraSetInteractor(std::shared_ptr<Scene> scene) : m_scene(scene) {}
CameraSetInteractor::~CameraSetInteractor() {}

void CameraSetInteractor::SetActiveCameraSet(std::shared_ptr<CameraSet> cameraSet) {}

std::vector<std::shared_ptr<Camera>> &CameraSetInteractor::GetCameras() {
    return m_cameraSet->GetCameras();
}

size_t CameraSetInteractor::GetAmountOfCameras() {
    return m_cameraSet->Size();
}

bool CameraSetInteractor::AreCamerasGenerated() {
    return m_cameraSet->AreCamerasGenerated();
}

bool CameraSetInteractor::IsCameraSetLocked() {
    return m_cameraSet->IsLocked();
}

bool CameraSetInteractor::LinkCameraSetToSceneObject(int id) {
    std::shared_ptr<SceneObject> obj = m_scene->Get(id);
    
    if(obj == nullptr){
        return false;
    }else{
        if(obj->GetType() == SceneObjectTypes::IMAGESET){
            m_cameraSet->LinkToImageSet(std::dynamic_pointer_cast<ImageSet>(obj), m_scene);
            return true;
        }
        return false;
    }
}
