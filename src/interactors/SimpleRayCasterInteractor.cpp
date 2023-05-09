#include <memory>
#include <glm/glm.hpp>
#include "../model/RayCaster/RayCaster.hpp"

#include "SimpleRayCasterInteractor.hpp"

using namespace glm;

SimpleRayCasterInteractor::SimpleRayCasterInteractor() {

}


SimpleRayCasterInteractor::~SimpleRayCasterInteractor() {

}

size_t SimpleRayCasterInteractor::GetAmountOfRays(){
    return m_rayCaster->GetAmountOfRays();
}

size_t SimpleRayCasterInteractor::GetRenderZoneWidth(){
    return m_rayCaster->GetRenderZoneWidth();
}

size_t SimpleRayCasterInteractor::GetRenderZoneHeight(){
    return m_rayCaster->GetRenderZoneHeight();
}

bool SimpleRayCasterInteractor::ShowRayLines(){
    return m_rayCaster->AreRaysVisible();
}

void SimpleRayCasterInteractor::SetShowRayLines(bool visible){
    m_rayCaster->SetRaysVisible(visible);
}

void SimpleRayCasterInteractor::SetActiveRayCaster(std::shared_ptr<RayCaster> rayCaster){
    m_rayCaster = rayCaster;
}