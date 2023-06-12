#include <memory>
#include <utility>
#include "NeRFInteractor.hpp"
#include "../model/Dataset/NeRFDataset.hpp"

void NeRFInteractor::SetNeRFDataset(std::shared_ptr<NeRFDataset> nerf)
{
    m_nerf = std::move(nerf);
}

bool NeRFInteractor::IsImageSetLoaded()
{
    std::shared_ptr<ImageSet> imageSet = m_nerf->GetImageSet();
    if(imageSet != nullptr){
        return imageSet->size() != 0;
    }
    return false;
}

int NeRFInteractor::GetImageSetId() {
    std::shared_ptr<ImageSet> imageSet = m_nerf->GetImageSet();
    if(imageSet != nullptr){
        return imageSet->GetID();
    }
    return -1;
}

bool NeRFInteractor::IsCalibrationLoaded() {
    return m_nerf->IsCalibrationLoaded();
}

// std::vector<Camera> &NeRFInteractor::GetCameras() {

//     return m_nerf->GetCameraSet()->GetCameras();
// }

const std::string& NeRFInteractor::GetCurrentJsonPath(){
    return m_nerf->GetCurrentJsonPath();
}

void NeRFInteractor::LoadCalibrations() {
    m_nerf->LoadCalibrations();
}

void NeRFInteractor::GenerateCameras() {
    m_nerf->GenerateCameras();
}


void NeRFInteractor::SetDatasetMode(size_t index){
    m_nerf->SetMode(NeRFDatasetModes(index));
}

bool NeRFInteractor::AreCamerasGenerated(){
    return m_nerf->AreCamerasGenerated();
}

void NeRFInteractor::LoadDataset(){
    m_nerf->Load();

}