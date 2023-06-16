//
// Created by hpiteau on 12/06/23.
//

#ifndef DRTMCS_DATALOADER_HPP
#define DRTMCS_DATALOADER_HPP


#include <memory>
#include <vector>
#include "../Camera/CameraSet.hpp"
#include "../ImageSet.hpp"
#include "Common.cuh"
#include "GPUData.cuh"

class DataLoader {
private:
    std::shared_ptr<CameraSet> m_cameraSet;
    std::shared_ptr<ImageSet> m_imageSet;

    unsigned int m_batchSize;
    /** Array of batch item descriptors. */
    std::vector<GPUData<BatchItemDescriptor>*> m_batchItems;
    /** Array of GPU-Ready pointers to gpu-allocated batch-items descriptors. */
    std::vector<BatchItemDescriptor*> m_gpuBatchItemPointers;

    /** A list of indexes that are used to select which cameras and images to
     * put in the batch. */
    std::vector<unsigned int> m_indexes;

    /** The index in the start of the batch indexes in the indexes array. */
    unsigned int m_startIndex{};

    bool m_isReady = false;
    bool m_batchLoaded = false;


public:
    DataLoader(std::shared_ptr<CameraSet> cameraSet ,std::shared_ptr<ImageSet> imageSet);
    DataLoader(const DataLoader&) = delete;
    ~DataLoader() = default;

    void Initialize();

    void SetBatchSize(unsigned int size);
    [[nodiscard]] unsigned int GetBatchSize() const;

    void Shuffle();

    void LoadNext();

    BatchItemDescriptor* GetGPUDescriptors();

    [[nodiscard]] bool IsReady() const;
    [[nodiscard]] bool IsOnGPU() const;

};


#endif //DRTMCS_DATALOADER_HPP
