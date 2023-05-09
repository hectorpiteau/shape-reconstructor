#include <string>

#include <string>
#include <iostream>
#include <filesystem>
#include <algorithm>

#include "ImageSet.hpp"
#include "../controllers/Scene/Scene.hpp"
#include "../../include/icons/IconsFontAwesome6.h"

ImageSet::ImageSet(Scene* scene) : SceneObject{std::string("IMAGESET"), SceneObjectTypes::IMAGESET}, m_folderPath("") {
    SetName(std::string(ICON_FA_IMAGES " ImageSet"));
    m_images = std::vector<Image*>();
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
    std::cout << "SET IMAGESET FOLDER PATH: " << path << std::endl;
}

const std::string& ImageSet::GetFolderPath(){
    return m_folderPath;
}

bool imageSort(Image* a, Image* b){
    return strcmp(a->filename.c_str(), b->filename.c_str()) > 0 ? false : true;
}
size_t ImageSet::LoadImages() {
    if(m_folderPath.length() <= 0) return 0;
    
    for (const auto & entry : std::filesystem::directory_iterator(m_folderPath)){
        std::cout << entry.path() << std::endl;
        Image* img = new Image(entry.path().filename());
        img->LoadPng(entry.path(), true, false);
        m_images.push_back(img);
    }
    /** sort by filename. */
    std::sort(m_images.begin(), m_images.end(), imageSort);
    return m_images.size();
}

int ImageSet::GetAmountOfImages() {
    return m_images.size();
}

Image* ImageSet::GetImage(int index) {
    if(index < m_images.size()) return m_images[index];
    return nullptr;
}

Image* ImageSet::GetImage(const std::string& filename) {
    for(int i=0; i<m_images.size(); ++i){
        if(strcmp(filename.c_str(), m_images[i]->filename.c_str()) == 0) return m_images[i];
    }
    return nullptr;
}

void ImageSet::Render(){
    for(auto& child : m_children){
        if(child->IsActive()) child->Render();
    }
}