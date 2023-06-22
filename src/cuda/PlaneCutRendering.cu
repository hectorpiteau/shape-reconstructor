/*
Author: Hector Piteau (hector.piteau@gmail.com)
PlaneCutRendering.cuh (c) 2023
Desc: PlaneCut rendering algorithms.
Created:  2023-04-13T12:33:22.433Z
Modified: 2023-04-25T12:53:31.894Z
*/

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>

#include <cuda_gl_interop.h>

#include "../utils/helper_cuda.h"
#include <device_launch_parameters.h>
#include <cmath>

#include "PlaneCutRendering.cuh"
#include "../model/RayCaster/Ray.h"
#include "SingleRayCaster.cuh"
#include "RayCasterParams.cuh"
#include "Common.cuh"
#include "GPUData.cuh"
#include "Utils.cuh"

using namespace glm;


__global__ void planeCutRendering(PlaneCutDescriptor *planeCut, CameraDescriptor *camera, VolumeDescriptor *volume) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= camera->width)
        return;
    if (y >= camera->height)
        return;

    Ray ray = SingleRayCaster::GetRay(vec2(camera->width - x, camera->height - y), camera);

    vec3 planeOrigin = vec3(0.0, 0.0, 0.0);
    vec3 planeNormal = vec3(0.0, 0.0, 0.0);

    switch (planeCut->axis) {
        case 0:
            planeOrigin.x = planeCut->pos;
            planeNormal.x = 1.0f;
            break;
        case 1:
            planeOrigin.y = planeCut->pos;
            planeNormal.y = 1.0f;
            break;
        case 2:
            planeOrigin.z = planeCut->pos;
            planeNormal.z = 1.0f;
            break;
    }

    vec3 intersection = VectorPlaneIntersection(ray.origin, ray.dir, planeOrigin, planeNormal);

    if (all(lessThan(intersection, planeCut->max)) && all(greaterThan(intersection, planeCut->min))) {
        vec4 res = ReadVolume(intersection, volume);
        res = abs(res);
//        res.x *= 1.0f;
//        res.y *= 255.0f;
//        res.z *= 1000.0f;
        res.w = 255.0f;
        res = clamp(res, vec4(0.0f), vec4(255.0f));
        surf2Dwrite<uchar4>(VEC4_TO_UCHAR4(res), planeCut->outSurface, x * sizeof(uchar4), y);
    } else {
        surf2Dwrite<uchar4>(make_uchar4(0, 0, 0, 0), planeCut->outSurface, x * sizeof(uchar4), y);
    }
}

extern "C"
void plane_cut_rendering_wrapper(GPUData<PlaneCutDescriptor> &planeCut, GPUData<VolumeDescriptor> &volume,
                                 GPUData<CameraDescriptor> &camera) {
    /** Max 1024 per block. As each pixel is independent, may be useful to search for optimal size. */
    dim3 threadsPerBlock(16, 16);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 numBlocks(
            (camera.Host()->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (camera.Host()->height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    /** Call the main rendering kernel. **/
    planeCutRendering<<<numBlocks, threadsPerBlock>>>(planeCut.Device(), camera.Device(), volume.Device());

    /** Get last error after rendering. */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(plane_cut_rendering_wrapper) ERROR: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}
