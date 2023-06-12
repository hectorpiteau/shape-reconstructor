#pragma once
#include <memory>
#include <vector>

#include "../model/Dataset/NeRFDataset.hpp"

class NeRFInteractor {
public:
    NeRFInteractor() = default;
    NeRFInteractor(const NeRFInteractor&) = delete;
    ~NeRFInteractor() = default;

    void SetNeRFDataset(std::shared_ptr<NeRFDataset> nerf);

    bool IsImageSetLoaded();

    int GetImageSetId();

    /**
     * @brief 
     * 
     * @return true 
     * @return false 
     */
    bool IsCalibrationLoaded();

    bool AreCamerasGenerated();

    /**
     * @brief Get the list of cameras related to each image.
     * 
     * @return std::vector<Camera>& 
     */
    // std::vector<Camera>& GetCameras();

    /**
     * @brief Get the Current Json Path string.
     * 
     * @return std::string& 
     */
    const std::string& GetCurrentJsonPath();

    /**
     * @brief Generate cameras for each images of the NeRF dataset. 
     * 
     */
    void GenerateCameras();

    /**
     * @brief 
     * 
     */
    void LoadCalibrations();

    void SetDatasetMode(size_t index);

    void LoadDataset();

private:
    std::shared_ptr<NeRFDataset> m_nerf;
};