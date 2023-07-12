#include <string>

#include <string>
#include <iostream>
// #include <experimental/filesystem>
#include "../utils/filesystem.h"

#include <algorithm>

#include "ImageSet.hpp"
#include "../controllers/Scene/Scene.hpp"
#include "../../include/icons/IconsFontAwesome6.h"

// using namespace std::experimental::filesystem::v1;

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
    /** If the paths are different, then the imageset is not loaded anymore. */
    if(std::strcmp(m_folderPath.c_str(), path.c_str()) != 0){
        m_isLoaded = false;
        UnloadImages();
    }

    m_folderPath = path;

    std::cout << "SET IMAGESET FOLDER PATH: " << path << std::endl;
}

void ImageSet::UnloadImages(){
    for(auto img : m_images){
        delete img;
    }
    m_images = std::vector<Image*>();
}

const std::string& ImageSet::GetFolderPath(){
    return m_folderPath;
}

bool imageSort(Image* a, Image* b){
    return strcmp(a->filename.c_str(), b->filename.c_str()) <= 0;
}

size_t ImageSet::LoadImages() {
    if(m_folderPath.length() <= 0) return 0;
    
    for (const auto & entry : std::filesystem::directory_iterator(m_folderPath)){
        auto* img = new Image(entry.path().filename());
        img->LoadPng(entry.path(), false, false);
        m_images.push_back(img);
    }
    /** sort by filename. */
    std::sort(m_images.begin(), m_images.end(), imageSort);

    m_isLoaded = true;
    return m_images.size();
}

Image* ImageSet::GetImage(size_t index) {
    if(index < m_images.size()) return m_images[index];
    return nullptr;
}

Image* ImageSet::GetImage(const std::string& filename) {
    for(auto & m_image : m_images){
        if(strcmp(filename.c_str(), m_image->filename.c_str()) == 0) return m_image;
    }
    return nullptr;
}

void ImageSet::Render(){
    for(auto& child : m_children){
        if(child->IsActive()) child->Render();
    }
}

bool ImageSet::IsLoaded() const {
    return m_isLoaded;
}
