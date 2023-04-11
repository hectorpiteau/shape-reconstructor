#pragma once 

#include "../view/SceneObject/SceneObject.hpp"
#include "../view/ImUI/InspectorView.hpp"

#include "ImageSetInteractor.hpp"
#include "CameraInteractor.hpp"

class InspectorView;

class SceneObjectInteractor {
public:
    SceneObjectInteractor();
    SceneObjectInteractor(const SceneObjectInteractor&) = delete;
    ~SceneObjectInteractor();

    void SetSelectedSceneObject(std::shared_ptr<SceneObject>& object);

    int GetSceneObjectId();

    const std::string& GetSceneObjectName();

    enum SceneObjectTypes GetSelectedSceneObjectType();
    
    void Render();

    ImageSetInteractor* imageSetInteractor;
    CameraInteractor* cameraInteractor;

private:
    std::shared_ptr<SceneObject> m_selectedSceneObject;

    InspectorView* m_inspectorView;
};