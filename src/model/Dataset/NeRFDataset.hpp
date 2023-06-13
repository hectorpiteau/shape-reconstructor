#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>

#include "Dataset.hpp" 
#include "../ImageSet.hpp"
#include "../../view/SceneObject/SceneObject.hpp"
#include "../../controllers/Scene/Scene.hpp"
#include "../Camera/CameraSet.hpp"
#include "../controllers/Scene/Scene.hpp"

using namespace glm;

enum NeRFDatasetModes {
    TRAIN,
    VALID
};

static const std::vector<const char*> NeRFDatasetModesNames = {
    "Train",
    "Valid"
};

struct NeRFImage : public CameraCalibrationInformations {
    /** Original transform matrix. Extrinsic + Intrinsic fused. */
    glm::mat4 transformMatrix;
    // glm::mat4 intrinsic;
    // glm::mat4 extrinsic;
    /** Full path to the file, including the filename. */
    std::string fullPath;
    /** Just the filename. */
    std::string fileName;
};

/**
 * @brief The NeRF Dataset using the lego model for now.
 * It can be used to load the dataset images descriptors. 
 * Real images can then be loaded by 
 */
class NeRFDataset : public Dataset, public SceneObject{
private:

    vec2 m_imageSize = vec2(800, 800);

    Scene* m_scene;

    enum NeRFDatasetModes m_mode;

    std::string m_trainJSONPath;
    std::string m_trainImagesPath;

    std::string m_validJSONPath;
    std::string m_validImagesPath;

    std::vector<NeRFImage> m_images;
    std::vector<CameraCalibrationInformations> m_imagesCalibration;

    bool m_isCalibrationLoaded;
    bool m_camerasGenerated;

    /** in dep */
    std::shared_ptr<CameraSet> m_cameraSet = nullptr;
    std::shared_ptr<ImageSet> m_imageSet = nullptr;

public:
    explicit NeRFDataset(Scene* scene);
    ~NeRFDataset() override;

    /**
     * @brief Load the images paths and transform matrices
     * in memory. 
     * Does not load the images themselves.
     * An ImageSet can be used for this task.
     * 
     * @return true 
     * @return false 
     */
    bool LoadCalibrations();

    /**
     * @brief Load images in the ImageSet.
     * 
     * @return true : Images loaded.
     * @return false : Images not loaded.
     */
    bool Load() override;

    /**
     * @brief Get the amount of images in the dataset currently loaded.
     * 
     * @return size_t : The amount of images.
     */
    size_t Size() override;

    /**
     * @brief Get the Mode of the Dataset.
     * 
     * @return enum NeRFDatasetModes : Either TRAIN, VALID, (TEST) 
     */
    enum NeRFDatasetModes GetMode();

    /**
     * @brief Get the Mode's name as a string (char*).
     * 
     * @return const char* : The mode's name.
     */
    const char* GetModeName();

    /**
     * @brief Set the Mode of the Dataset, 
     * will change the dataset used.
     * 
     * @param mode : Either TRAIN or VALID.
     */
    void SetMode(enum NeRFDatasetModes mode);

    void Render() override;

    const std::string& GetCurrentJsonPath();
    const std::string& GetCurrentImageFolderPath();



    [[nodiscard]] bool IsCalibrationLoaded() const;
    
    void GenerateCameras();

    [[nodiscard]] bool AreCamerasGenerated() const;

    std::shared_ptr<CameraSet> GetCameraSet() override;
    std::shared_ptr<ImageSet> GetImageSet() override;
};