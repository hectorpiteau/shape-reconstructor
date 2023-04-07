#include "ImageSet.hpp"
#include <string>

#include <string>
#include <iostream>
#include <filesystem>

#include "../../include/icons/IconsFontAwesome6.h"

ImageSet::ImageSet() : SceneObject {std::string("IMAGESET")}, m_folderPath(""), m_images(0) {
    SetName(std::string(ICON_FA_IMAGES " ImageSet"));
}


void ImageSet::SetFolderPath(const std::string& path){
    m_folderPath = path;
}

void ImageSet::LoadImages() {
    if(m_folderPath.length() <= 0) return;
    
    for (const auto & entry : std::filesystem::directory_iterator(m_folderPath)){
        std::cout << entry.path() << std::endl;
        Image* img = new Image();
        img->LoadPng(entry.path(), false, false);
        m_images.push_back(img);
    }
}

int ImageSet::GetAmountOfImages() {
    return m_images.size();
}

const Image* ImageSet::GetImage(int index) {
    if(index < m_images.size()) return m_images[index];
}

void ImageSet::Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene){

}