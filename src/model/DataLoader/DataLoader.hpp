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
#include "../Dataset/Dataset.hpp"

class DataLoader {
private:
    /** The dataset to use. It defines how to recover each image.*/
    std::shared_ptr<Dataset> m_dataset;

    /** The amount if images in each batch. */
    unsigned int m_batchSize;

    /** Array of batch item descriptors. */
    std::vector<GPUData<BatchItemDescriptor>*> m_batchItems;

    std::vector<CudaBuffer<vec3>*> m_losses;
    std::vector<CudaBuffer<vec3>*> m_cpreds;


    /** A list of indexes that are used to select which cameras and images to
     * put in the batch. */
    std::vector<unsigned int> m_indexes;

    /** The index in the start of the batch indexes in the indexes array. */
    unsigned int m_startIndex{};

    bool m_isReady = false;
    bool m_batchLoaded = false;

public:
    explicit DataLoader(std::shared_ptr<Dataset> dataset);
    DataLoader(const DataLoader&) = delete;
    ~DataLoader() = default;

    void Initialize();

    void SetBatchSize(unsigned int size);
    [[nodiscard]] unsigned int GetBatchSize() const;

    void Shuffle();
    /**
     * Load the current batch descriptors in memory.
     */
    void LoadBatch();

    /**
     * Go to the next batch indexes.
     */
    void NextBatch();

    /**
     * Unload and unmap resources of the current batch.
     */
    void UnloadBatch();

    std::vector<GPUData<BatchItemDescriptor>*> GetGPUDatas();

    [[nodiscard]] bool IsReady() const;
    [[nodiscard]] bool IsOnGPU() const;

};


#endif //DRTMCS_DATALOADER_HPP
