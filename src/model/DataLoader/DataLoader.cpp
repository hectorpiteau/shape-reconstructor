//
// Created by hpiteau on 12/06/23.
//

#include "DataLoader.hpp"
#include "../Dataset/Dataset.hpp"
#include <algorithm>
#include <utility>
#include <vector>
#include <random>

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset)
        : m_dataset(std::move(dataset)), m_batchSize(5) {
    /** Allocate */
    m_batchItems = std::vector<GPUData<BatchItemDescriptor> *>(m_batchSize);
    m_losses = std::vector<CudaBuffer<vec4>*>(m_batchSize);
    m_cpreds = std::vector<CudaBuffer<vec4>*>(m_batchSize);

    for (size_t i = 0; i < m_batchSize; ++i) {
        auto tmp = new GPUData<BatchItemDescriptor>();
        m_batchItems[i] = tmp;

        auto loss_buff = new CudaBuffer<vec4>();
        auto cpred_buff = new CudaBuffer<vec4>();

        m_losses[i] = loss_buff;
        m_cpreds[i] = cpred_buff;
    }
}

void DataLoader::Initialize() {
    /** Initialize and shuffle indexes. */
    m_startIndex = 0;
    m_indexes = std::vector<unsigned  int>();
    for (unsigned int i = 0; i < m_dataset->Size(); i++) m_indexes.push_back(i);
    Shuffle();
    m_isReady = true;

    for (size_t i = 0; i < m_batchSize; ++i) {
        m_losses[i]->Allocate(m_dataset->GetImageSet()->GetImage(0)->width * m_dataset->GetImageSet()->GetImage(0)->height);
        m_cpreds[i]->Allocate(m_dataset->GetImageSet()->GetImage(0)->width * m_dataset->GetImageSet()->GetImage(0)->height);
    }
}

bool DataLoader::IsReady() const {
    return m_isReady;
}

bool DataLoader::IsOnGPU() const {
    return m_batchLoaded;
}

void DataLoader::Shuffle() {
    auto rd = std::random_device{};
    auto rng = std::default_random_engine{rd()};
    std::shuffle(std::begin(m_indexes), std::end(m_indexes), rng);
}

void DataLoader::LoadBatch(RenderMode mode) {
    if (!m_isReady || !m_dataset->GetCameraSet()->AreCamerasGenerated() || !m_dataset->GetImageSet()->IsLoaded()) return;


    /** Load new batch */
    for (unsigned int i = 0; i < m_batchSize; ++i) {
        auto index = m_indexes[m_startIndex + i];
        auto res = m_dataset->GetEntry(index);

        m_batchItems[i]->Host()->cam = res.cam->GetGPUDescriptor();
        m_batchItems[i]->Host()->img = res.img->GetGPUDescriptor();
        m_batchItems[i]->Host()->loss = m_losses[i]->Device();
        m_batchItems[i]->Host()->cpred = m_cpreds[i]->Device();
        m_batchItems[i]->Host()->res = ivec2(res.img->width, res.img->height);
        m_batchItems[i]->Host()->range = res.cam->GetIntegrationRangeGPUDescriptor().Device();
        m_batchItems[i]->Host()->debugRender = true;
        m_batchItems[i]->Host()->mode = mode;
        m_batchItems[i]->Host()->debugSurface = res.cam->GetCudaTexture()->OpenSurface();
        m_batchItems[i]->ToDevice();
    }

    m_batchLoaded = true;
}

void DataLoader::NextBatch() {
    /** Increment the start_index or set it to 0 if it cannot generate a full batch. */
    if (m_startIndex + m_batchSize >= m_indexes.size()) {
        Shuffle();
        m_startIndex = 0;
    } else {
        m_startIndex += m_batchSize;
    };
}

void DataLoader::UnloadBatch() {
    for (unsigned int i = 0; i < m_batchSize; ++i) {
        auto camera = m_dataset->GetCameraSet()->GetCamera(m_indexes[m_startIndex + i]);
        camera->GetCudaTexture()->CloseSurface();
        /** Unload previously loaded images in memory. */
//        auto image = m_dataset->GetImageSet()->GetImage(m_indexes[m_startIndex + i]);
//        image->FreeGPUDescriptor();
    }
    m_batchLoaded = false;
}

void DataLoader::SetBatchSize(unsigned int size) {
    if (size != m_batchSize) {
        /** Unload images. */
        if (m_batchLoaded) {
            for (unsigned int i = 0; i < m_batchSize; ++i) {
                auto image = m_dataset->GetImageSet()->GetImage(m_indexes[m_startIndex + i]);
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

std::vector<GPUData<BatchItemDescriptor>*> DataLoader::GetGPUDatas() {
    return m_batchItems;
}