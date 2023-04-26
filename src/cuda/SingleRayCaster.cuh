/*
Author: Hector Piteau (hector.piteau@gmail.com)
SingleRayCaster.hpp (c) 2023
Desc: Cast one ray per pixel.
Created:  2023-04-14T11:30:44.487Z
Modified: 2023-04-17T11:39:56.003Z
*/

#ifndef SINGLE_RAY_CASTER_H
#define SINGLE_RAY_CASTER_H

#include "../model/RayCaster/Ray.h"
#include "CudaRayCaster.cuh"
#include <glm/glm.hpp>
#include "Projection.cuh"
#include "RayCasterParams.cuh"

using namespace glm;



class SingleRayCaster
{
private:
    
public:
    SingleRayCaster(){};

    ~SingleRayCaster() {};
    
    __device__ static Ray GetRay(const glm::vec2 &pixel, RayCasterParams params){
        vec3 dir = vec3(0.0);
        /** Compute ray from camera to pixel. Undistort. */
        dir = PixelToWorld(pixel, params.intrinsic, params.extrinsic, params.width, params.height);

        Ray ray = {
            .origin = params.worldPos,
            .dir = dir,
            .tmin = 0,
            .tmax = 1};

        return ray;
    }
};

#endif //SINGLE_RAY_CASTER_H