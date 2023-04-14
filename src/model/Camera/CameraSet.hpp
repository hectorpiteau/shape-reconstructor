#pragma once

#include "Camera.hpp"
#include "../ImageSet.hpp"
#include "../../controllers/Scene/Scene.hpp"
#include "../view/SceneObject/SceneObject.hpp"
#include <vector>
#include <string>
#include <glm/glm.hpp>

struct CameraCalibrationInformations {
    glm::mat4 intrisic;
    glm::mat4 extrinsic;

    float fov;
};

struct CameraEntry {
    std::shared_ptr<Camera> camera;
    std::shared_ptr<Image> image;

    std::string filename;

    bool isCalibrated;
};

/**
 * @brief This class is used to represent a set of cameras. 
 * It can be used 
 * 
 */
class CameraSet : public SceneObject {
public:
    CameraSet();
    CameraSet(const CameraSet& ) = delete;
    ~CameraSet();   

    /**
     * @brief Add a camera to the CameraSet. 
     * 
     * @param camera : A shared ptr to a camera. 
     */
    void AddCamera(std::shared_ptr<Camera> camera);

    /**
     * @brief Get the Size of the camera set. 
     * How many cameras in the set.
     * 
     * @return size_t : A positive integer.
     */
    size_t Size();

    /**
     * @brief Operator [] in order to easily index cameras by the index 
     * in the vector.
     * 
     * @param index : The index in the vector. Must be smaller than size.
     * @return std::shared_ptr<Camera>& : A shared ptr of the camera if it exists.
     */
    std::shared_ptr<Camera>& operator[](size_t index);
    
    /**
     * @brief Get a camera by its id.
     * 
     * @param id : The id of the desired camera.
     * @return std::shared_ptr<Camera> : The camera if it exist, nullptr otherwise.
     */
    std::shared_ptr<Camera> GetCameraById(size_t id);

    /**
     * @brief Get the list of all the cameras. 
     * 
     * @return std::vector<std::shared_ptr<Camera>>& : A ref to the vector of cameras.  
     */
    std::vector<std::shared_ptr<Camera>>& GetCameras();

    bool LinkToImageSet(std::shared_ptr<ImageSet> imageSet, std::shared_ptr<Scene> scene);
    
    bool AreCamerasGenerated();

    bool IsLocked();

    bool CalibrateFromInformations(const std::vector<struct CameraCalibrationInformations>& informations);

    void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene);

private:
    std::vector<std::shared_ptr<Camera>> m_cameras;

    bool m_areCameraGenerated;
    bool m_areCalibrated;

    bool m_isLocked;
};