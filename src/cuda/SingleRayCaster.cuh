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
#include "Common.cuh"

using namespace glm;



class SingleRayCaster
{
private:
    
public:
    SingleRayCaster(){};

    ~SingleRayCaster() {};
    
    __device__ static Ray GetRay(const glm::vec2 &pixel, CameraDescriptor* camera){
        vec3 dir = vec3(0.0);
        /** Compute ray from camera to pixel. Undistort. */
        dir = PixelToWorld(pixel, camera->camInt, camera->camExt, camera->width, camera->height);
        dir = dir - camera->camPos;
        Ray ray = {
            .origin = camera->camPos,
            .dir =  dir,
            .tmin = 0.0f,
            .tmax = 20.0f
        };

        return ray;
    }
};

#endif //SINGLE_RAY_CASTER_H