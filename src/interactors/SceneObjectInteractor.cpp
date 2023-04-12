#include <memory>

#include "SceneObjectInteractor.hpp"

#include "ImageSetInteractor.hpp"
#include "CameraInteractor.hpp"
#include "NeRFInteractor.hpp"
#include "CameraSetInteractor.hpp"

#include "../model/ImageSet.hpp"
#include "../model/Dataset/NeRFDataset.hpp"
#include "../view/ImUI/InspectorView.hpp"

SceneObjectInteractor::SceneObjectInteractor()
{
    imageSetInteractor = new ImageSetInteractor();
    cameraInteractor = new CameraInteractor();
    nerfInteractor = new NeRFInteractor();
    cameraSetInteractor = new CameraSetInteractor();

    m_inspectorView = new InspectorView(this);
}

SceneObjectInteractor::~SceneObjectInteractor(){
    delete m_inspectorView;

    delete imageSetInteractor;
    delete cameraInteractor;
    delete nerfInteractor;
    delete cameraSetInteractor;
}

void SceneObjectInteractor::SetSelectedSceneObject(std::shared_ptr<SceneObject> &object)
{
    m_selectedSceneObject = object;

    switch(object->GetType()){
        case SceneObjectTypes::IMAGESET:
            imageSetInteractor->SetActiveImageSet(std::dynamic_pointer_cast<ImageSet>(object));
            break;
        case SceneObjectTypes::CAMERA:
            cameraInteractor->SetCamera(std::dynamic_pointer_cast<Camera>(object));
            break;
        case SceneObjectTypes::NERFDATASET:
            nerfInteractor->SetNeRFDataset(std::dynamic_pointer_cast<NeRFDataset>(object));
            break;
        case SceneObjectTypes::CAMERASET:
            cameraSetInteractor->SetActiveCameraSet(std::dynamic_pointer_cast<CameraSet>(object));
            break;
    }
}

void SceneObjectInteractor::Render()
{
    m_inspectorView->Render();
}

int SceneObjectInteractor::GetSceneObjectId()
{
    return m_selectedSceneObject->GetID();
}

const std::string &SceneObjectInteractor::GetSceneObjectName()
{
    return m_selectedSceneObject->GetName();
}

enum SceneObjectTypes SceneObjectInteractor::GetSelectedSceneObjectType(){
    if(m_selectedSceneObject == nullptr) return SceneObjectTypes::NONE;
    return m_selectedSceneObject->GetType();
}