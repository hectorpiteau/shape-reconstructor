#include <string>
#include <memory>
#include <fstream>
#include <nlohmann/json.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "NeRFDataset.hpp"
#include "../../include/icons/IconsFontAwesome6.h"

#include "NeRFDataset.hpp"
#include "../ImageSet.hpp"
#include "../Camera/CameraSet.hpp"

#include "../../controllers/Scene/Scene.hpp"
#include "../../include/icons/IconsFontAwesome6.h"

using json = nlohmann::json;
using namespace glm;

NeRFDataset::NeRFDataset(Scene *scene,
                         const std::string trainJson,
                         const std::string trainImages,
                         const std::string validJson,
                         const std::string validImages)
        : Dataset{std::string("NeRFDataset")}, SceneObject{std::string("NeRFDataset"), SceneObjectTypes::NERFDATASET},
          m_scene(scene),
          m_mode(NeRFDatasetModes::TRAIN),
          m_trainJSONPath(trainJson),
          m_trainImagesPath(trainImages),
          m_validJSONPath(validJson),
          m_validImagesPath(validImages),
          m_images(),
          m_imagesCalibration(),
          m_entries(),
          m_isCalibrationLoaded(false){
    SetName(std::string(ICON_FA_DATABASE " Nerf Dataset"));
    m_children = std::vector<std::shared_ptr<SceneObject>>();

    /** Create the image_set. */
    m_imageSet = std::make_shared<ImageSet>(m_scene);
    m_scene->Add(m_imageSet, true, true);
    m_children.push_back(m_imageSet);
    m_imageSet->SetFolderPath(GetCurrentImageFolderPath());

    /** Create a CameraSet */
    m_cameraSet = std::make_shared<CameraSet>(m_scene);
    m_scene->Add(m_cameraSet, true, true);
    m_children.push_back(m_cameraSet);
}

NeRFDataset::~NeRFDataset() = default;

bool NeRFDataset::LoadCalibrations() {
    std::string configFilePath = m_mode == NeRFDatasetModes::TRAIN ? m_trainJSONPath : m_validJSONPath;
    std::cout << "Load calibration file: " << configFilePath << std::endl;

    std::ifstream f(configFilePath);
    json data = json::parse(f);
    float fov = 0.69f;

    if (data["frames"] == nullptr || data["frames"].is_array() == false) {
        f.close();
        m_isCalibrationLoaded = false;
        return m_isCalibrationLoaded;
    }

    /** Parse FOV x : */
    if (data["camera_angle_x"] != nullptr && data["camera_angle_x"].is_number_float()) {
        fov = data["camera_angle_x"];
    }

    int image_counter = 0;
    for (auto &img: data["frames"]) {
        /** Verify that transform_matrix exists and that it is an array. */
        if (img["transform_matrix"] == nullptr || img["transform_matrix"].is_array() == false) {
            continue;
        }

        struct NeRFImage tmp = {};
        tmp.fov = fov;

        /** Copy path and filename. */
        if (img["file_path"] != nullptr && img["file_path"].is_string()) {
            tmp.fullPath = std::string(img["file_path"]);
            tmp.fullPath.erase(0, 1);
            tmp.fullPath = std::string("../data/nerf" + tmp.fullPath);

            /** Copy file name. (last element in the string). */
            static const std::string delimiter = "/";
            std::string str = img["file_path"];
            size_t last = 0, next = 0;
            while ((next = str.find(delimiter, last)) != std::string::npos)
                last = next + 1;
            tmp.fileName = str.substr(last) + std::string(".png");
            std::cout << "Load file: " << tmp.fileName << std::endl;
        }

        /** Copy matrix columns. */
        for (int i = 0; i < 4; i++) {
            /** Verify the correctness of the i'th column in the json. */
            if (img["transform_matrix"][i] == nullptr || img["transform_matrix"][i].is_array() == false) {
                break;
            }

            /** Copy matrix transform in the mat4. */
            for (int j = 0; j < 4; j++) {
                /** Verify that the correctness of the float. */
                if (img["transform_matrix"][i][j] == nullptr ||
                    img["transform_matrix"][i][j].is_number_float() == false) {
                    break;
                }

                float value = img["transform_matrix"][i][j];
                tmp.transformMatrix[i][j] = value;
            }
        }

        tmp.transformMatrix = glm::transpose(tmp.transformMatrix);
        tmp.transformMatrix = glm::inverse(tmp.transformMatrix);
        /** negate row 1 */
        tmp.transformMatrix[0][1] = -tmp.transformMatrix[0][1];
        tmp.transformMatrix[1][1] = -tmp.transformMatrix[1][1];
        tmp.transformMatrix[2][1] = -tmp.transformMatrix[2][1];
        tmp.transformMatrix[3][1] = -tmp.transformMatrix[3][1];

        /** negate row 2 */
        tmp.transformMatrix[0][2] = -tmp.transformMatrix[0][2];
        tmp.transformMatrix[1][2] = -tmp.transformMatrix[1][2];
        tmp.transformMatrix[2][2] = -tmp.transformMatrix[2][2];
        tmp.transformMatrix[3][2] = -tmp.transformMatrix[3][2];

//        tmp.transformMatrix[0][3] = -tmp.transformMatrix[0][3];
//        tmp.transformMatrix[1][3] = -tmp.transformMatrix[1][3];
//        tmp.transformMatrix[2][3] = -tmp.transformMatrix[2][3];
//        tmp.transformMatrix[3][3] = -tmp.transformMatrix[3][3];

        /** Create intrinsic and extrinsic matrices. */
        mat4 K = mat4(1.0f);
        float fx = m_imageSize.x / (2.0 * tan(tmp.fov / 2.0));
        float fy = m_imageSize.y / (2.0 * tan(tmp.fov / 2.0));
        K[0][0] = fx;
        K[1][1] = fy;
        K[2][0] = m_imageSize.x / 2.0;
        K[2][1] = m_imageSize.y / 2.0;
        K[2][2] =  1.0f;
        K[3][3] =  1.0f;

        tmp.intrinsic = K;
        tmp.extrinsic = tmp.transformMatrix;
        // mat3x3 R = mat3x3(tmp.extrinsic);
        // vec4 T = vec4(tmp.extrinsic[3]);

        tmp.extrinsic = glm::rotate(tmp.extrinsic, glm::half_pi<float>(), glm::vec3(1.0, 0.0, 0.0));

        m_images.push_back(tmp);

        CameraCalibrationInformations calib = {.intrinsic = tmp.intrinsic, .extrinsic = tmp.extrinsic, .fov = tmp.fov};
        m_imagesCalibration.push_back(calib);

        image_counter += 1;
    }

    m_isCalibrationLoaded = true;
    return m_isCalibrationLoaded;
}

size_t NeRFDataset::Size()
{
    return m_images.size();
}

const char *NeRFDataset::GetModeName()
{
    switch (m_mode)
    {
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

void NeRFDataset::SetMode(enum NeRFDatasetModes mode) {
    m_mode = mode;
    m_cameraSet->Reset();

    m_isCalibrationLoaded = false;

    switch (mode) {
        case NeRFDatasetModes::TRAIN:
            GetImageSet()->SetFolderPath(m_trainImagesPath);
            break;
        case NeRFDatasetModes::VALID:
            GetImageSet()->SetFolderPath(m_validImagesPath);
            break;
        default:
            GetImageSet()->SetFolderPath(m_trainImagesPath);
            break;
    }
}

void NeRFDataset::Render() {
    m_imageSet->Render();
    m_cameraSet->Render();
}

const std::string &NeRFDataset::GetCurrentJsonPath() {
    switch (m_mode) {
        case NeRFDatasetModes::TRAIN:
            return m_trainJSONPath;
        case NeRFDatasetModes::VALID:
            return m_validJSONPath;
        default:
            return m_trainJSONPath;
    }
}

const std::string &NeRFDataset::GetCurrentImageFolderPath() {
    switch (m_mode) {
        case NeRFDatasetModes::TRAIN:
            return m_trainImagesPath;
        case NeRFDatasetModes::VALID:
            return m_validImagesPath;
        default:
            return m_trainImagesPath;
    }
}


bool NeRFDataset::IsCalibrationLoaded() const {
    return m_isCalibrationLoaded;
}

bool NeRFDataset::Load() {
    /** Load images in the ImageSet with the path. */
    m_imageSet->LoadImages();
    LoadCalibrations();
    GenerateCameras();
    return m_cameraSet->AreCamerasGenerated() && m_isCalibrationLoaded;
}

void NeRFDataset::GenerateCameras() {
    if (m_cameraSet->AreCamerasGenerated()) {
        std::cout << "NeRFDataset::GenerateCameras : Cameras already generate." << std::endl;
        return;
    }
    if (!m_isCalibrationLoaded) {
        std::cout << "NeRFDataset::GenerateCameras : Calibration not done yet, can't generate cameras." << std::endl;
        return;
    }

    std::cout << "NeRFDataset::GenerateCameras : Generate Cameras. " << std::endl;

    int cpt = 0;

    for (size_t i = 0; i < m_imagesCalibration.size(); ++i) {
        NeRFImage imgInfo = m_images[i];
        CameraCalibrationInformations calibInfo = m_imagesCalibration[i];

        std::shared_ptr<Camera> cam = std::make_shared<Camera>(m_scene);

        auto img = m_imageSet->GetImage(imgInfo.fileName);
        cam->SetIntrinsic(calibInfo.intrinsic);
        cam->SetExtrinsic(calibInfo.extrinsic);
        cam->SetImage(img);
        cam->SetResolution(ivec2(m_imageSet->GetImage(imgInfo.fileName)->width, m_imageSet->GetImage(imgInfo.fileName)->height));
        cam->SetActive(true);
        cam->SetIsChild(true);
        cam->SetName(std::string(ICON_FA_CAMERA " Camera ") + std::to_string(cpt++));
        cam->SetIsVisibleInList(false);
        cam->InitializeCudaTexture();

        m_scene->Add(cam, true, true);

        m_entries.push_back({.cam=cam, .img=img});
        m_cameraSet->AddCamera(cam);
    }

    m_cameraSet->SetCameraGenerated(true);
    m_cameraSet->SetIsLocked(true);


}

bool NeRFDataset::AreCamerasGenerated() const {
    return m_cameraSet->AreCamerasGenerated();
}

std::shared_ptr<CameraSet> NeRFDataset::GetCameraSet() {
    return m_cameraSet;
}

std::shared_ptr<ImageSet> NeRFDataset::GetImageSet() {
    return m_imageSet;
}

DatasetEntry NeRFDataset::GetEntry(size_t index) {
    if(index >= Size()) return {};
    return m_entries[index];
}

void NeRFDataset::SetSourcePath(const std::string &train_path, const std::string &valid_path) {
    m_trainImagesPath = train_path;
    m_validImagesPath = valid_path;
    m_imageSet->SetFolderPath(GetCurrentImageFolderPath());

    m_cameraSet->Reset();
    m_images = std::vector<NeRFImage>();
    m_entries = std::vector<DatasetEntry>();
    m_imagesCalibration = std::vector<CameraCalibrationInformations>();

    Load();
}
