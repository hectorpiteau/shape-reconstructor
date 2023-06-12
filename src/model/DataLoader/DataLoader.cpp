//
// Created by hpiteau on 12/06/23.
//

#include "DataLoader.hpp"

DataLoader::DataLoader(std::shared_ptr<CameraSet> cameraSet, std::shared_ptr<ImageSet> imageSet)
: m_cameraSet(std::move(cameraSet)), m_imageSet(std::move(imageSet)), m_batchSize(10) {

}

void DataLoader::SetBatchSize(unsigned int size) {
    m_batchSize = size;
}

unsigned int DataLoader::GetBatchSize() {
    return m_batchSize;
}
