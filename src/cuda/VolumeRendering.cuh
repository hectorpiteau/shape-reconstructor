/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRendering.cuh (c) 2023
Desc: Volume rendering algorithms.
Created:  2023-04-13T12:33:22.433Z
Modified: 2023-04-26T12:26:19.942Z
*/

#ifndef VOLUME_RENDERING_H
#define VOLUME_RENDERING_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <glm/glm.hpp>

#include "../model/RayCaster/Ray.h"
#include "RayCasterParams.cuh"
using namespace glm;

struct VolumeData
{
    float4 data;
};

// Forward declaration of CUDA render
extern "C" void volume_rendering_wrapper(RayCasterParams& params, cudaTextureObject_t &volume, float4 *outTexture, size_t width, size_t height);

extern "C" void volume_rendering_wrapper_linear(RayCasterParams& params, float4* volume, float4 *outTexture, size_t width, size_t height);

// void volume_rendering_wrapper(cudaTextureObject_t& volume, cudaSurfaceObject_t& outTexture, size_t width, size_t height, size_t depth);
#endif // VOLUME_RENDERING_H
