//
// Created by hpiteau on 12/06/23.
//

#include "DataLoader.hpp"
#include <algorithm>
#include <vector>
#include <random>

DataLoader::DataLoader()
: m_cameraSet(nullptr), m_imageSet(nullptr), m_batchSize(10), m_gpuBatchItemPointers(), m_indexes() {
    /** Allocate */
    m_batchItems = (GPUData<BatchItemDescriptor>*)malloc(m_batchSize * sizeof(GPUData<BatchItemDescriptor>));


}

void DataLoader::Init(){
    if(!m_isReady) return;
    m_startIndex = 0;
    for(unsigned int i=0; i < m_cameraSet->Size(); i++) m_indexes.push_back(i);
    Shuffle();
    LoadNext();
}

bool DataLoader::IsReady() const {
    return m_isReady;
}

bool DataLoader::IsOnGPU() const {
    return m_batchLoaded;
}

void DataLoader::Shuffle(){
    auto rd = std::random_device {};
    auto rng = std::default_random_engine { rd() };
    std::shuffle(std::begin(m_indexes), std::end(m_indexes), rng);
}

void DataLoader::LoadNext(){
    if(!m_isReady || !m_cameraSet->AreCamerasGenerated() || !m_imageSet->AreImagesGenerated()) return;
    /** Unload current batch. */
    if(m_batchLoaded){
        /** Unload previously loaded images in memory. */
        for(unsigned int i=0; i < m_batchSize; ++i){
            auto image = m_imageSet->GetImage(m_indexes[m_startIndex + i]);
            image->FreeGPUDescriptor();
        }
    }
    m_gpuBatchItemPointers.clear();

    /** Increment the start_index or set it to 0 if it cannot generate a full batch. */
    if(m_startIndex+m_batchSize >= m_indexes.size()){
        Shuffle();
        m_startIndex = 0;
    }else{
        m_startIndex += m_batchSize;
    };


    /** Load new batch */
    for(unsigned int i=0; i < m_batchSize; ++i){
        auto camera = m_cameraSet->GetCamera(m_indexes[m_startIndex +  i]);
        auto cameraDesc = camera->GetGPUDescriptor();

        auto image = m_imageSet->GetImage(m_indexes[m_startIndex + i]);
        auto imageDesc = image->GetGPUDescriptor();

        auto integrationRangeDesc = camera->GetIntegrationRangeGPUDescriptor().Device();

        m_batchItems[i].Host()->cam = cameraDesc;
        m_batchItems[i].Host()->img = imageDesc;
        m_batchItems[i].Host()->range = integrationRangeDesc;
        m_batchItems[i].ToDevice();

        /** Store ready to use pointer array. */
        m_gpuBatchItemPointers.push_back(m_batchItems[i].Device());
    }

    m_batchLoaded = true;
}

void DataLoader::SetBatchSize(unsigned int size) {
    if(size != m_batchSize){
        /** Unload images. */
        if(m_batchLoaded){
            for(unsigned int i=0; i < m_batchSize; ++i){
                auto image = m_imageSet->GetImage(m_indexes[m_startIndex + i]);
                image->FreeGPUDescriptor();
            }
        }

        free(m_batchItems);
        m_batchSize = size;
        m_batchItems = (GPUData<BatchItemDescriptor>*)malloc(m_batchSize * sizeof(GPUData<BatchItemDescriptor>));
    }
}

unsigned int DataLoader::GetBatchSize() const {
    return m_batchSize;
}

BatchItemDescriptor* DataLoader::GetGPUDescriptors(){
    return m_gpuBatchItemPointers[0];
}

void DataLoader::SetCameraSet(std::shared_ptr<CameraSet> cameraSet) {
    m_cameraSet = std::move(cameraSet);
    m_isReady = (m_cameraSet != nullptr) && (m_imageSet != nullptr);
    Init();

}

void DataLoader::SetImageSet(std::shared_ptr<ImageSet> imageSet) {
    m_imageSet = std::move(imageSet);
    m_isReady = (m_cameraSet != nullptr) && (m_imageSet != nullptr);
    Init();
}
