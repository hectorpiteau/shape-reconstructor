#pragma once

#include <memory>

#include "../model/Volume3D.hpp"

class Volume3DInteractor {
public:

Volume3DInteractor();
Volume3DInteractor(const Volume3DInteractor&) = delete
~Volume3DInteractor();

void SetActiveVolume3D(std::shared_ptr<Volume3D> volume);

std::shared_ptr<Volume3D>& GetVolume3D();

glm::vec3 GetResolution(){
    return m_volume->

}

void SetResolution(){

}


private:
    std::shared_ptr<Volume3D> m_volume;
    
};