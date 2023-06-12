#pragma once 

#include "../view/SceneObject/SceneObject.hpp"
#include "../view/ImUI/InspectorView.hpp"

#include "../controllers/Scene/Scene.hpp"

#include "ImageSetInteractor.hpp"
#include "CameraInteractor.hpp"
#include "NeRFInteractor.hpp"
#include "AdamInteractor.hpp"
#include "CameraSetInteractor.hpp"
#include "Volume3DInteractor.hpp"
#include "PlaneCutInteractor.hpp"
#include "VolumeRendererInteractor.hpp"
#include "SimpleRayCasterInteractor.hpp"

class InspectorView;

/**
 * @brief The SceneObjectInteractor can be used to interact 
 * with any SceneObject. 
 * It is the main entry point for any SceneObject Editor (Inspector). 
 * This interactor is communicating with all other specific interactor in
 * order to update them with the selected object. 
 * This update will allow the ui to be rendered using the interactor and it's 
 * newly refreshed information.
 * 
 */
class SceneObjectInteractor {
public:
    SceneObjectInteractor(Scene* m_scene);
    SceneObjectInteractor(const SceneObjectInteractor&) = delete;
    ~SceneObjectInteractor();

    /**
     * @brief Set the Selected SceneObject in the scene.
     * 
     * @param object : Any SceneObject in the scene.
     */
    void SetSelectedSceneObject(std::shared_ptr<SceneObject>& object);

    /**
     * @brief Get the selected SceneObject's id.
     * 
     * @return int : The id of the selected SceneObject.
     */
    int GetSceneObjectId();

    /**
     * @brief Get the selected SceneObject's name.
     * 
     * @return const std::string& : The name of the selected SceneObject.
     */
    const std::string& GetSceneObjectName();

    /**
     * @brief Get the Selected SceneObject type.
     * 
     * @return enum SceneObjectTypes : The type of the selected SceneObject.
     */
    enum SceneObjectTypes GetSelectedSceneObjectType();
    
    /**
     * @brief Call rendering processes of the view.
     */
    void Render();

    ImageSetInteractor* imageSetInteractor;
    CameraInteractor* cameraInteractor;
    NeRFInteractor* nerfInteractor;
    CameraSetInteractor* cameraSetInteractor;
    Volume3DInteractor* volume3DInteractor;
    VolumeRendererInteractor* volumeRendererInteractor;
    SimpleRayCasterInteractor* simpleRayCasterInteractor;
    PlaneCutInteractor* planeCutInteractor;
    AdamInteractor* adamInteractor;
private:
    Scene* m_scene;
    /** The currently selected SceneObject. */
    std::shared_ptr<SceneObject> m_selectedSceneObject;

    /** The main window that holds all other SceneObject's editors (inspectors). */
    InspectorView* m_inspectorView;

};