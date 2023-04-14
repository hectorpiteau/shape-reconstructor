#pragma once
#include <string>
#include <memory>

#include "../model/Camera/CameraSet.hpp"
#include "../model/Camera/Camera.hpp"

#include "../controllers/Scene/Scene.hpp"

/**
 * @brief Interactor that is used to interact and edit a
 * CameraSet Object.
 */
class CameraSetInteractor {
public:
    CameraSetInteractor(std::shared_ptr<Scene> scene);
    CameraSetInteractor(const CameraSetInteractor &) = delete;
    ~CameraSetInteractor();

    
    /**
     * @brief Set the currently active CameraSet object for this 
     * interactor.
     * 
     * @param cameraSet : A shared pointer to a CameraSet.
     */
    void SetActiveCameraSet(std::shared_ptr<CameraSet> cameraSet);

    /**
     * @brief Get the vector of cameras associated with the cameraSet.
     * 
     * @return std::vector<std::shared_ptr<Camera>>& : A vector of shared_ptr of cameras. 
     */
    std::vector<std::shared_ptr<Camera>>& GetCameras();

    /**
     * @brief Get the Amount Of Cameras in the set.
     * 
     * @return size_t : The amount >= 0. 
     */
    size_t GetAmountOfCameras();

    /**
     * @brief Checks if the cameras are generated or not.
     * 
     * @return true : The cameras are generated.
     * @return false : The cameras are not generated.
     */
    bool AreCamerasGenerated();

    /**
     * @brief Check if the CameraSet is locked to another SceneObject.
     * For example to an ImageSet. 
     * 
     * @return true : The CameraSet is locked.
     * @return false : The CameraSet is not locked.
     */
    bool IsCameraSetLocked();

    /**
     * @brief Link the CameraSet to another SceneObject.
     * Can be used to link the CameraSet to an ImageSet in order to
     * represent each image an a camera input view.
     * 
     * @param id
     * @return true : in case of a success. 
     * @return false : in case of a failure (obj does not exist). 
     */
    bool LinkCameraSetToSceneObject(int id);

private:
    std::shared_ptr<Scene> m_scene;

    std::shared_ptr<CameraSet> m_cameraSet;
};