#include <memory>

#include "SceneObjectInteractor.hpp"

#include "../model/PlaneCut.hpp"
#include "ImageSetInteractor.hpp"
#include "CameraInteractor.hpp"
#include "NeRFInteractor.hpp"
#include "CameraSetInteractor.hpp"
#include "PlaneCutInteractor.hpp"
#include "AdamInteractor.hpp"
#include "../model/Camera/CameraSet.hpp"
#include "../model/ImageSet.hpp"
#include "../model/Dataset/NeRFDataset.hpp"
#include "../view/ImUI/InspectorView.hpp"
#include "../controllers/Scene/Scene.hpp"

SceneObjectInteractor::SceneObjectInteractor(Scene* scene)
: m_scene(scene)
{
    imageSetInteractor = new ImageSetInteractor();
    cameraInteractor = new CameraInteractor();
    nerfInteractor = new NeRFInteractor();
    cameraSetInteractor = new CameraSetInteractor(m_scene);
    volumeRendererInteractor = new VolumeRendererInteractor(m_scene);
    volume3DInteractor = new Volume3DInteractor();
    simpleRayCasterInteractor = new SimpleRayCasterInteractor();
    planeCutInteractor = new PlaneCutInteractor(m_scene);
    adamInteractor = new AdamInteractor();

    m_inspectorView = new InspectorView(this);
}

SceneObjectInteractor::~SceneObjectInteractor(){
    /** Delete the View Object generated. */
    delete m_inspectorView;

    /** Delete dependencies interactors. */
    delete imageSetInteractor;
    delete cameraInteractor;
    delete nerfInteractor;
    delete cameraSetInteractor;
    delete simpleRayCasterInteractor;
    delete planeCutInteractor;
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
        case SceneObjectTypes::VOLUME3D:
            volume3DInteractor->SetActiveVolume3D(std::dynamic_pointer_cast<DenseVolume3D>(object));
            break;
        case SceneObjectTypes::VOLUMERENDERER:
            volumeRendererInteractor->SetCurrentVolumeRenderer(std::dynamic_pointer_cast<VolumeRenderer>(object));
            break;
        case SceneObjectTypes::RAYCASTER:
            simpleRayCasterInteractor->SetActiveRayCaster(std::dynamic_pointer_cast<RayCaster>(object));
            break;
        case SceneObjectTypes::CAMERASET:
            cameraSetInteractor->SetActiveCameraSet(std::dynamic_pointer_cast<CameraSet>(object));
            break;
        case SceneObjectTypes::PLANECUT:
            planeCutInteractor->SetActivePlaneCut(std::dynamic_pointer_cast<PlaneCut>(object));
            break;
        case SceneObjectTypes::ADAMOPTIMIZER:
            adamInteractor->SetAdamOptimizer(std::dynamic_pointer_cast<AdamOptimizer>(object));
            break;
        case SceneObjectTypes::LINEGRID:
        case SceneObjectTypes::GIZMO:
        case SceneObjectTypes::MODEL:
        case SceneObjectTypes::MESH:
        case SceneObjectTypes::NONE:
        default:
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