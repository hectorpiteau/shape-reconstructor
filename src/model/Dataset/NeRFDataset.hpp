#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>

#include "Dataset.hpp" 
#include "../ImageSet.hpp"
#include "../../view/SceneObject/SceneObject.hpp"
#include "../../controllers/Scene/Scene.hpp"


enum NeRFDatasetModes {
    TRAIN,
    VALID
};

static const std::vector<const char*> NeRFDatasetModesNames = {
    "Train",
    "Valid"
};

struct NeRFImage {
    /** Extrinsic + Intrinsic fused. */
    glm::mat4 transformMatrix;
    /** Full path to the file, including the filename. */
    std::string fullPath;
    /** Just the filename. */
    std::string fileName;

    float fov;
};

/**
 * @brief The NeRF Dataset using the lego model for now.
 * It can be used to load the dataset images descriptors. 
 * Real images can then be loaded by 
 */
class NeRFDataset : public Dataset, public SceneObject{
public:
    NeRFDataset(std::shared_ptr<Scene> scene, std::shared_ptr<ImageSet> imageSet);
    ~NeRFDataset();

    /**
     * @brief Load the images paths and transform matrices
     * in memory. 
     * Does not load the images itselfs. 
     * An ImageSet can be used for this task.
     * 
     * @return true 
     * @return false 
     */
    bool Load();

    size_t Size();

    /**
     * @brief Get the Mode of the Dataset.
     * 
     * @return enum NeRFDatasetModes : Either TRAIN, VALID, (TEST) 
     */
    enum NeRFDatasetModes GetMode();
    const char* GetModeName();

    /**
     * @brief Set the Mode of the Dataset, 
     * will change the dataset used.
     * 
     * @param mode : Either TRAIN or VALID.
     */
    void SetMode(enum NeRFDatasetModes mode);

    void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene);

    const std::string& GetCurrentJsonPath();
    const std::string& GetCurrentImageFolderPath();

    std::shared_ptr<ImageSet> GetImageSet();

    bool IsCalibrationLoaded();

    void LoadCalibrations();
    void GenerateCameras();

    bool AreCamerasGenerated();

private:
    enum NeRFDatasetModes m_mode;
    
    std::string m_trainJSONPath;
    std::string m_trainImagesPath;

    std::string m_validJSONPath;
    std::string m_validImagesPath;

    std::vector<struct NeRFImage> m_images;
    std::shared_ptr<Scene> m_scene;

    bool m_isCalibrationLoaded;
    bool m_camerasGenerated;
};