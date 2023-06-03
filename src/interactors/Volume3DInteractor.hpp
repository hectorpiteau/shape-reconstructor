
#pragma once

#include <memory>
#include <glm/glm.hpp>

#include "../model/Volume3D.hpp"

using namespace glm;

class Volume3DInteractor
{
public:
    Volume3DInteractor();
    Volume3DInteractor(const Volume3DInteractor &) = delete ;
    ~Volume3DInteractor();

    void SetActiveVolume3D(std::shared_ptr<Volume3D> volume);

    std::shared_ptr<Volume3D> &GetVolume3D();

    const ivec3 &GetResolution();
    void SetResolution(const ivec3 &resolution);

    bool IsRenderingZoneVisible();
    void SetIsRenderingZoneVisible(bool visible);

    const vec3& GetBboxMin();
    const vec3& GetBboxMax();

    void SetBboxMin(const vec3& min);
    void SetBboxMax(const vec3& max);

    const vec3* GetBBox();

private:
    std::shared_ptr<Volume3D> m_volume;

    bool m_isRenderingZoneVisible = false;
};