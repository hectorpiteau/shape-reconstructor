/*
Author: Hector Piteau (hector.piteau@gmail.com)
RayCaster.hpp (c) 2023
Desc: Ray caster is used to define rays outgoing from a camera.
Created:  2023-04-14T09:50:13.297Z
Modified: 2023-04-17T09:20:25.454Z
*/
#pragma once

#include <memory>
#include "Ray.h"
#include "../Camera/Camera.hpp"

#include "../../cuda/VolumeRendering.cuh"

using namespace glm;

class RayCaster : public CudaRayCaster
{
protected:
    std::shared_ptr<Camera> m_camera;
public:
    RayCaster(std::shared_ptr<Camera> camera) : m_camera(camera) {};
    ~RayCaster() {}
    RayCaster(const RayCaster&) = delete;

    void SetCamera(std::shared_ptr<Camera> camera) { m_camera = camera;}

    virtual Ray GetRay(const vec2& pixel) = 0;
};



