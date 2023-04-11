#pragma once

#include "Camera.hpp"
#include <vector>

/**
 * @brief This class is used to represent a set of cameras. 
 * It can be used 
 * 
 */
class CameraSet {
public:
    CameraSet();
    CameraSet(const CameraSet& ) = delete;

    void AddCamera(std::shared_ptr<Camera> camera);

    size_t Length();

    ~CameraSet();

    std::shared_ptr<Camera>& operator[](size_t index);
    
    /**
     * @brief Get a camera by its id.
     * 
     * @param id : The id of the desired camera.
     * @return std::shared_ptr<Camera> : The camera if it exist, nullptr otherwise.
     */
    std::shared_ptr<Camera> GetCameraById(size_t id);

private:
    std::vector<std::shared_ptr<Camera>> m_cameras;
};