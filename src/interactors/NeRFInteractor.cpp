#include <memory>
#include "NeRFInteractor.hpp"
#include "../model/Dataset/NeRFDataset.hpp"

NeRFInteractor::NeRFInteractor()
{

}

NeRFInteractor::~NeRFInteractor()
{
}

void NeRFInteractor::SetNeRFDataset(std::shared_ptr<NeRFDataset> nerf)
{
    m_nerf = nerf;
}

bool NeRFInteractor::IsImageSetLoaded()
{
    std::shared_ptr<ImageSet> imgset = m_nerf->GetImageSet();
    if(imgset != nullptr){
        return imgset->GetAmountOfImages() == 0 ? false : true;
    }
    return false;
}

int NeRFInteractor::GetImageSetId() {
    std::shared_ptr<ImageSet> imgset = m_nerf->GetImageSet();
    if(imgset != nullptr){
        return imgset->GetID();
    }
    return -1;
}

bool NeRFInteractor::IsCalibrationLoaded() {
    return m_nerf->IsCalibrationLoaded();
}

std::vector<Camera> &NeRFInteractor::GetCameras() {
    
}

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