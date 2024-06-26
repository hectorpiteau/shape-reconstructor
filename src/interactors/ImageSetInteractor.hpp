#pragma once

#include <memory>

#include "../model/ImageSet.hpp"

class ImageSetInteractor {
public:

ImageSetInteractor();
ImageSetInteractor(const ImageSetInteractor&) = delete;

~ImageSetInteractor();

void SetActiveImageSet(std::shared_ptr<ImageSet> imageSet);

size_t LoadImages(const char* folderPath);

std::shared_ptr<ImageSet>& GetImageSet();

[[nodiscard]] bool GetUpdatedImageSet() const;

void SetUpdatedImageSet(bool value);

private:
    std::shared_ptr<ImageSet> m_imageSet;

    bool m_updatedImageSet = true;
    
};