#include <string>
#include <memory>
#include <fstream>
#include <nlohmann/json.hpp>
#include <glm/glm.hpp>

#include "NeRFDataset.hpp"
#include "../ImageSet.hpp"
#include "../Camera/CameraSet.hpp"

#include "../../controllers/Scene/Scene.hpp"
#include "../../include/icons/IconsFontAwesome6.h"

using json = nlohmann::json;

NeRFDataset::NeRFDataset(std::shared_ptr<Scene> scene, std::shared_ptr<ImageSet> imageSet)
    : SceneObject{std::string("NeRFDataset"), SceneObjectTypes::NERFDATASET}, Dataset(std::string("NeRFDataset")), m_scene(scene), 
    m_trainJSONPath("../data/nerf/transforms_train.json"), 
    m_validJSONPath("../data/nerf/transforms_val.json"),
    m_trainImagesPath("../data/nerf/train/"),
    m_validImagesPath("../data/nerf/val/")
{
    SetName(std::string(ICON_FA_DATABASE " Nerf Dataset"));

    m_children = std::vector<std::shared_ptr<SceneObject>>();
    m_children.push_back(imageSet);
    imageSet->SetFolderPath(GetCurrentImageFolderPath());

    m_isCalibrationLoaded = false;
    m_camerasGenerated = false;
}

NeRFDataset::~NeRFDataset()
{
}

bool NeRFDataset::Load()
{
    std::string configFilePath = m_mode == NeRFDatasetModes::TRAIN ? m_trainJSONPath : m_validJSONPath;
    std::cout << "Load calibration file: " << configFilePath << std::endl;

    std::ifstream f(configFilePath);
    json data = json::parse(f);
    float fov = 0.69f; 

    if (data["frames"] == nullptr || data["frames"].is_array() == false)
    {
        f.close();
        return false;
    }

    /** Parse FOV x : */
    if(data["camera_angle_x"] != nullptr && data["camera_angle_x"].is_number_float()){
        fov = data["camera_angle_x"];
    }

    int image_counter = 0;
    for (auto &img : data["frames"])
    {
        /** Verify that transform_matrix exists and that it is an array. */
        if (img["transform_matrix"] == nullptr || img["transform_matrix"].is_array() == false)
        {
            continue;
        }

        struct NeRFImage tmp = {};
        tmp.fov = fov;

        /** Copy path and filename. */
        if (img["file_path"] != nullptr && img["file_path"].is_string())
        {
            tmp.fullPath = std::string(img["file_path"]);
            tmp.fullPath.erase(0, 1);
            tmp.fullPath = std::string("../data/nerf" + tmp.fullPath);

            /** Copy file name. (last element in the string). */
            static const std::string delimiter = "/";
            std::string str = img["file_path"];
            size_t last = 0, next = 0;
            while ((next = str.find(delimiter, last)) != std::string::npos)
                last = next + 1;
            tmp.fileName = str.substr(last);
            std::cout << "Load file: " << tmp.fileName << std::endl;
        }

        /** Copy matrix columns. */
        for (int i = 0; i < 4; i++)
        {
            /** Verify the correctness of the i'th column in the json. */
            if (img["transform_matrix"][i] == nullptr || img["transform_matrix"][i].is_array() == false)
            {
                break;
            }

            /** Copy matrix transform in the glm::mat4. */
            for (int j = 0; j < 4; j++)
            {
                /** Verify that the correctness of the float. */
                if (img["transform_matrix"][i][j] == nullptr || img["transform_matrix"][i][j].is_number_float() == false)
                {
                    break;
                }

                float value = img["transform_matrix"][i][j];
                tmp.transformMatrix[i][j] = value;
            }

            /** Create intrinsic and extrinsic matrices. */
            glm::mat4 K = glm::mat4(1.0f);
            float fx = width / (2.0 * glm::tan(tmp.fov / 2.0));
            float fy = height / (2.0 * glm::tan(tmp.fov / 2.0));
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = width / 2.0
            K[1, 2] = height / 2.0
            tmp.intrinsic = K;

            


        }

        m_images.push_back(tmp);
        image_counter += 1;
    }
    return true;
}

size_t NeRFDataset::Size()
{
    return m_images.size();
}

const char* NeRFDataset::GetModeName(){
    switch(m_mode){
        case NeRFDatasetModes::TRAIN:
            return NeRFDatasetModesNames[0];
        case NeRFDatasetModes::VALID:
            return NeRFDatasetModesNames[1];
        default:
            return NeRFDatasetModesNames[0];
    }
}
enum NeRFDatasetModes NeRFDataset::GetMode()
{
    return m_mode;
}


void NeRFDataset::SetMode(enum NeRFDatasetModes mode)
{
    m_mode = mode;
    m_camerasGenerated = false;
    m_isCalibrationLoaded = false;

    switch (mode)
    {
    case NeRFDatasetModes::TRAIN:
        GetImageSet()->SetFolderPath(m_trainImagesPath);
        break;
    case NeRFDatasetModes::VALID:
        GetImageSet()->SetFolderPath(m_validImagesPath);
        break;
    }
}

void NeRFDataset::Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene)
{
    // nothing for now
}

const std::string& NeRFDataset::GetCurrentJsonPath(){
    switch(m_mode){
        case NeRFDatasetModes::TRAIN:
            return m_trainJSONPath;
        case NeRFDatasetModes::VALID:
            return m_validJSONPath;
        default:
            return m_trainJSONPath;
    }
}

const std::string& NeRFDataset::GetCurrentImageFolderPath(){
    switch(m_mode){
        case NeRFDatasetModes::TRAIN:
            return m_trainImagesPath;
        case NeRFDatasetModes::VALID:
            return m_validImagesPath;
        default:
            return m_trainImagesPath;
    }
}


std::shared_ptr<ImageSet> NeRFDataset::GetImageSet(){
    if(m_children[0]->GetType() != SceneObjectTypes::IMAGESET) 
        return std::shared_ptr<ImageSet>(nullptr);
    else
        return std::dynamic_pointer_cast<ImageSet>(m_children[0]);
}

bool NeRFDataset::IsCalibrationLoaded(){
    return m_isCalibrationLoaded;
}

void NeRFDataset::LoadCalibrations(){
    m_isCalibrationLoaded = Load();
}

void NeRFDataset::GenerateCameras(){
    /** Create a CameraSet */
    std::shared_ptr<CameraSet> camSet = std::make_shared<CameraSet>();
    
    m_scene->Add(camSet, true, true);
    
    m_children.push_back(camSet);

    camSet->LinkToImageSet(GetImageSet(), m_scene);

    camSet->CalibrateFromInformations(m_images);

    m_camerasGenerated = true;
}

bool NeRFDataset::AreCamerasGenerated(){
    return m_camerasGenerated;
}