#include "NeRFDataset.hpp"
#include <string>

#include <fstream>
#include <nlohmann/json.hpp>
#include <glm/glm.hpp>

using json = nlohmann::json;

NeRFDataset::NeRFDataset() : Dataset(std::string("NeRFDataset")),
                             m_trainJSONPath("../data/nerf/transforms_train.json"), m_validJSONPath("../data/nerf/transforms_val.json")
{
}

NeRFDataset::~NeRFDataset()
{
}

bool NeRFDataset::Load()
{
    std::string configFilePath = m_mode == NeRFDatasetModes::TRAIN ? m_trainJSONPath : m_validJSONPath;

    std::ifstream f(configFilePath);
    json data = json::parse(f);

    if (data["frames"] == nullptr || data["frames"].is_array() == false)
    {
        f.close();
        return false;
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
        }

        m_images.push_back(tmp);
        image_counter += 1;
    }
}

enum NeRFDatasetModes NeRFDataset::GetMode()
{
    return m_mode;
}

void NeRFDataset::SetMode(enum NeRFDatasetModes mode)
{
    m_mode = mode;
}