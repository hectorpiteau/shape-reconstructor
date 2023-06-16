/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRendering.cuh (c) 2023
Desc: Volume rendering algorithms.
Created:  2023-04-13T12:33:22.433Z
Modified: 2023-04-25T12:53:31.894Z
*/

#ifndef VOLUME_RENDERING_H
#define VOLUME_RENDERING_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <glm/glm.hpp>

#include "../model/RayCaster/Ray.h"
#include "RayCasterParams.cuh"
#include "Common.cuh"
#include "GPUData.cuh"

using namespace glm;

/**
 * @brief Volume Rendering Wrapper using Linear Memory Allocation :
 
 */
extern "C" void volume_rendering_wrapper(GPUData<RayCasterDescriptor>& raycaster, GPUData<CameraDescriptor>& camera, GPUData<VolumeDescriptor>& volume, cudaSurfaceObject_t surface);

#endif // VOLUME_RENDERING_H