#pragma once

#include <memory>

#include "../model/ImageSet.hpp"

class Volume3DInteractor {
public:

Volume3DInteractor();
Volume3DInteractor(const Volume3DInteractor&) = delete
~Volume3DInteractor();

std::shared_ptr<ImageSet>& GetImageSet();

private:
    std::shared_ptr<ImageSet> m_imageSet;
    
};