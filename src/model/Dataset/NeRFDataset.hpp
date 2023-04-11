#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>

#include "Dataset.hpp" 

enum NeRFDatasetModes {
    TRAIN,
    VALID
};

struct NeRFImage {
    /** Extrinsic + Intrinsic fused. */
    glm::mat4 transformMatrix;
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
class NeRFDataset : public Dataset{
public:
    NeRFDataset();
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

    /**
     * @brief Get the Mode of the Dataset.
     * 
     * @return enum NeRFDatasetModes : Either TRAIN, VALID, (TEST) 
     */
    enum NeRFDatasetModes GetMode();

    /**
     * @brief Set the Mode of the Dataset, 
     * will change the dataset used.
     * 
     * @param mode : Either TRAIN or VALID.
     */
    void SetMode(enum NeRFDatasetModes mode);

private:
    enum NeRFDatasetModes m_mode;
    std::string m_trainJSONPath;
    std::string m_validJSONPath;

    std::vector<struct NeRFImage> m_images;
};