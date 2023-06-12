//
// Created by hpiteau on 12/06/23.
//

#ifndef DRTMCS_DATALOADER_HPP
#define DRTMCS_DATALOADER_HPP


#include <memory>
#include "../Camera/CameraSet.hpp"
#include "../ImageSet.hpp"
#include "Common.cuh"
#include "GPUData.cuh"

class DataLoader {
private:
    std::shared_ptr<CameraSet> m_cameraSet;
    std::shared_ptr<ImageSet> m_imageSet;

    unsigned int m_batchSize;

    GPUData<BatchDescriptor> m_batchDescriptor;

public:
    explicit DataLoader(std::shared_ptr<CameraSet> cameraSet, std::shared_ptr<ImageSet> imageSet);
    DataLoader(const DataLoader&) = delete;
    ~DataLoader() = default;

    void SetBatchSize(unsigned int size);
    unsigned int GetBatchSize();

};


#endif //DRTMCS_DATALOADER_HPP
