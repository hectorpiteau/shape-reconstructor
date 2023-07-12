#pragma once
#include <string>
#include <utility>
#include "../Camera/CameraSet.hpp"


struct DatasetEntry {
    std::shared_ptr<Camera> cam;
    Image* img;
};

/**
 * @brief 
 * 
 */
class Dataset {
public:

explicit Dataset(std::string name) : m_name(std::move(name)) {

};

const std::string& GetName(){
    return m_name;
}

/**
 * @brief Load the dataset files in host memory. 
 * 
 * @return true : Loaded with success.
 * @return false : Not loaded.
 */
virtual bool Load() = 0;

virtual size_t Size() = 0;

virtual std::shared_ptr<CameraSet> GetCameraSet() = 0;
virtual std::shared_ptr<ImageSet> GetImageSet() = 0;

virtual DatasetEntry GetEntry(size_t index) = 0;

virtual void SetSourcePath(const std::string &train_path, const std::string &valid_path) = 0;

private:
    std::string m_name;
};