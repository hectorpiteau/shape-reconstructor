//
// Created by hpiteau on 12/06/23.
//

#include "DataLoader.hpp"
#include <algorithm>
#include <utility>
#include <vector>
#include <random>

DataLoader::DataLoader(std::shared_ptr<CameraSet> cameraSet ,std::shared_ptr<ImageSet> imageSet)
: m_cameraSet(std::move(cameraSet)), m_imageSet(std::move(imageSet)), m_batchSize(10), m_gpuBatchItemPointers(){
    /** Allocate */
    m_batchItems = std::vector<GPUData<BatchItemDescriptor>*>(m_batchSize);
    for (int i = 0; i < m_batchSize; ++i) {
        auto tmp = new GPUData<BatchItemDescriptor>();
        m_batchItems[i] = tmp;
    }
}

void DataLoader::Initialize(){
    /** Initialize and shuffle indexes. */
    m_startIndex = 0;
    for(unsigned int i=0; i < m_cameraSet->Size(); i++) m_indexes.push_back(i);
    Shuffle();
    m_isReady = true;
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

    /** Load new batch */
    for(unsigned int i=0; i < m_batchSize; ++i){
//        std::cout << "DATASET Load Batch: index: " << std::to_string(m_startIndex +  i) << ", cam_id: " << std::to_string(m_indexes[m_startIndex +  i]) << std::endl;
        std::cout << "DATASET: "<< std::to_string(i) << std::endl;
        auto camera = m_cameraSet->GetCamera(m_indexes[m_startIndex +  i]);
        auto cameraDesc = camera->GetGPUDescriptor();

        auto image = m_imageSet->GetImage(m_indexes[m_startIndex + i]);
        auto imageDesc = image->GetGPUDescriptor();

        auto integrationRangeDesc = camera->GetIntegrationRangeGPUDescriptor().Device();

        m_batchItems[i]->Host()->cam = cameraDesc;
        m_batchItems[i]->Host()->img = imageDesc;
        m_batchItems[i]->Host()->range = integrationRangeDesc;
        m_batchItems[i]->Host()->debugRender = false;
        m_batchItems[i]->ToDevice();

        /** Store ready to use pointer array. */
        m_gpuBatchItemPointers.push_back(m_batchItems[i]->Device());
    }

    /** Increment the start_index or set it to 0 if it cannot generate a full batch. */
    if(m_startIndex+m_batchSize >= m_indexes.size()){
        Shuffle();
        m_startIndex = 0;
    }else{
        m_startIndex += m_batchSize;
    };

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

//        free(m_batchItems);
//        m_batchSize = size;
//        m_batchItems = (GPUData<BatchItemDescriptor>*)malloc(m_batchSize * sizeof(GPUData<BatchItemDescriptor>));
    }
}

unsigned int DataLoader::GetBatchSize() const {
    return m_batchSize;
}

BatchItemDescriptor* DataLoader::GetGPUDescriptors(){
    return m_gpuBatchItemPointers[0];
}