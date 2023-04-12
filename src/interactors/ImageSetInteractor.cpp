#include <memory>

#include "../model/ImageSet.hpp"
#include "ImageSetInteractor.hpp"

ImageSetInteractor::ImageSetInteractor()
{

};

ImageSetInteractor::~ImageSetInteractor(){

};


void ImageSetInteractor::SetActiveImageSet(std::shared_ptr<ImageSet> imageSet){
    m_imageSet = imageSet;
    SetUpdatedImageSet(true);
}

std::shared_ptr<ImageSet>& ImageSetInteractor::GetImageSet(){
    return m_imageSet;
}

size_t ImageSetInteractor::LoadImages(const char* folderPath){
    m_imageSet->SetFolderPath(folderPath);
    return m_imageSet->LoadImages();

}

bool ImageSetInteractor::GetUpdatedImageSet(){return m_updatedImageSet;}

void ImageSetInteractor::SetUpdatedImageSet(bool value){
    m_updatedImageSet = value;
}