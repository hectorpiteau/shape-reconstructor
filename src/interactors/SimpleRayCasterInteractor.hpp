#pragma once
#include <memory>
#include <glm/glm.hpp>

#include "../model/RayCaster/RayCaster.hpp"

using namespace glm;

class SimpleRayCasterInteractor
{
private:
    std::shared_ptr<RayCaster> m_rayCaster;
public:
    SimpleRayCasterInteractor();
    SimpleRayCasterInteractor(const SimpleRayCasterInteractor &) = delete ;
    ~SimpleRayCasterInteractor();

    bool ShowRayLines();
    void SetShowRayLines(bool visible);

    size_t GetAmountOfRays();
    size_t GetRenderZoneWidth();
    size_t GetRenderZoneHeight();

    void SetActiveRayCaster(std::shared_ptr<RayCaster> rayCaster);

};