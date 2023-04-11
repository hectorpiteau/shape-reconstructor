#include <string>

#include <string>
#include <iostream>
#include <filesystem>

#include "ImageSet.hpp"
#include "../../include/icons/IconsFontAwesome6.h"

ImageSet::ImageSet() : SceneObject{std::string("IMAGESET"), SceneObjectTypes::IMAGESET}, m_folderPath("") {
    SetName(std::string(ICON_FA_IMAGES " ImageSet"));
    
    m_images = std::vector<Image*>();
    // m_images.reserve(100);

}

std::vector<Image *>& ImageSet::GetImages(){
    return m_images;
}

const Image* ImageSet::operator[](size_t index){
    return m_images[index];
}

size_t ImageSet::size(){
    return m_images.size();
}

void ImageSet::SetFolderPath(const std::string& path){
    m_folderPath = path;
}

size_t ImageSet::LoadImages() {
    if(m_folderPath.length() <= 0) return 0;
    
    for (const auto & entry : std::filesystem::directory_iterator(m_folderPath)){
        std::cout << entry.path() << std::endl;
        Image* img = new Image(entry.path().filename());
        img->LoadPng(entry.path(), false, false);
        m_images.push_back(img);
    }
    return m_images.size();
}

int ImageSet::GetAmountOfImages() {
    return m_images.size();
}

const Image* ImageSet::GetImage(int index) {
    if(index < m_images.size()) return m_images[index];
    return nullptr;
}

void ImageSet::Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene){

}