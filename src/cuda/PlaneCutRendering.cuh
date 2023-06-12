/*
Author: Hector Piteau (hector.piteau@gmail.com)
PlaneCutRendering.cuh (c) 2023
Desc: PlaneCut rendering algorithms.
Created:  2023-04-13T12:33:22.433Z
Modified: 2023-04-25T12:53:31.894Z
*/

#ifndef PLANE_CUT_RENDERING_H
#define PLANE_CUT_RENDERING_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <glm/glm.hpp>
#include "Common.cuh"
#include "GPUData.cuh"

using namespace glm;

extern "C" void plane_cut_rendering_wrapper(GPUData<PlaneCutDescriptor>& planeCut, GPUData<VolumeDescriptor>& volume, GPUData<CameraDescriptor>& camera);

#endif // PLANE_CUT_RENDERING_H