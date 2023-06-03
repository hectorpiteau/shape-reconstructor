#include <memory>
#include <glm/glm.hpp>

#include "../model/Volume3D.hpp"

#include "Volume3DInteractor.hpp"

using namespace glm;

Volume3DInteractor::Volume3DInteractor()
{
}

Volume3DInteractor::~Volume3DInteractor()
{
}

void Volume3DInteractor::SetActiveVolume3D(std::shared_ptr<Volume3D> volume)
{
    m_volume = volume;
}

std::shared_ptr<Volume3D> &Volume3DInteractor::GetVolume3D()
{
    return m_volume;
}

const ivec3 &Volume3DInteractor::GetResolution()
{
    return m_volume->GetResolution();
}

void Volume3DInteractor::SetResolution(const ivec3 &resolution)
{
}


bool Volume3DInteractor::IsRenderingZoneVisible(){
    return m_isRenderingZoneVisible;
}

void Volume3DInteractor::SetIsRenderingZoneVisible(bool visible){
    m_isRenderingZoneVisible = visible;
}

const vec3& Volume3DInteractor::GetBboxMin(){
    return m_volume->GetBboxMin();
}

const vec3& Volume3DInteractor::GetBboxMax(){
    return m_volume->GetBboxMax();
}

void Volume3DInteractor::SetBboxMin(const vec3& min){
    m_volume->SetBBoxMin(min);
}

void Volume3DInteractor::SetBboxMax(const vec3& max){
    m_volume->SetBBoxMax(max);
}

const vec3* Volume3DInteractor::GetBBox(){
    return m_volume->m_bboxPoints;
}