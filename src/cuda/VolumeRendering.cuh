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
using namespace glm;

struct VolumeData
{
    float4 data;
};

/**
 * @brief Volume Rendering Wrapper using Texture Allocation :
 * volume : cuda Texture3D
 * outTex : float4
 */
// extern "C" void volume_rendering_wrapper(RayCasterParams& params, cudaTextureObject_t &volume, float4 *outTexture, size_t width, size_t height);

/**
 * @brief Volume Rendering Wrapper using Linear Memory Allocation :
 * volume : float4
 * outTex : float4
 */
// extern "C" void volume_rendering_wrapper_linear(RayCasterParams& params, float4* volume, float4 *outTexture, size_t width, size_t height);

/**
 * @brief Volume Rendering Wrapper using Linear Memory Allocation :
 
 */
extern "C" void volume_rendering_wrapper_linea_ui8(RayCasterDescriptor* raycaster, CameraDescriptor* camera, VolumeDescriptor* volume);

#endif // VOLUME_RENDERING_H