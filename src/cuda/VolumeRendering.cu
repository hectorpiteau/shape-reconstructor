/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRendering.cu (c) 2023
Desc: Volume rendering algorithms.
Created:  2023-04-13T12:33:22.433Z
Modified: 2023-05-11T22:28:51.324Z
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

#include "VolumeRendering.cuh"
#include "../model/RayCaster/Ray.h"
#include "SingleRayCaster.cuh"

#include "Utils.cuh"
#include "Common.cuh"
#include "GPUData.cuh"

using namespace glm;


__device__ float tsdfToAlpha(float tsdf, float previousTsdf, float density) {
    if (previousTsdf > tsdf) {
        return (
                       1.0f + exp(-density * previousTsdf)) /
               (1.0f + exp(-density * tsdf));
    } else {
        return 1.0f;
    }
}


__device__ bool IsPointInBBox(const vec3 &point, VolumeDescriptor *volume) {
    if (all(lessThan(point, volume->bboxMax)) && all(greaterThan(point, volume->bboxMin)))
        return true;
    else
        return false;
}

__device__ short IsPointInVolume(const vec3 &point) {
    if (any(lessThan(point, vec3(-0.5, -0.5, -0.5))) || any(greaterThan(point, vec3(0.5, 0.5, 0.5))))
        return 0;
    return 1;
}

__device__ vec4 forward(Ray &ray, VolumeDescriptor *volume) //, float4* volume, const ivec3& resolution)
{
    /** Partial transmittance. */
    float Tpartial = 1.0f;
    /** Partial color. */
    vec3 Cpartial = vec3(0.0f, 0.0f, 0.0f);

    float step = 0.01f;

    /** The ray's min must be strictly smaller than max. */
    if (ray.tmin < ray.tmax) {

        /** Travel through the ray from it's min to max. */
        for (float t = ray.tmin; t < ray.tmax; t += step) {
            vec3 pos = ray.origin + t * ray.dir;

            if (IsPointInBBox(pos, volume)) {
                vec4 data = ReadVolume(pos, volume);
                vec3 color = vec3(data.r, data.g, data.b);
                float alpha = data.a;

                Cpartial += Tpartial * (1 - exp(-alpha)) * color;
//                Cpartial += Tpartial * alpha * color;
//                Tpartial *= (1.0f - alpha);
                Tpartial *= (1.0f / exp(alpha));

                if (Tpartial < 0.001f) {
                    Tpartial = 0.0f;
                    break;
                }
            }
        }
    }
    return {Cpartial, Tpartial};
}


__global__ void volumeRenderingUI8(RayCasterDescriptor *raycaster, CameraDescriptor *camera, VolumeDescriptor *volume) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= camera->width || y >= camera->height) return;

    if (!raycaster->renderAllPixels) {
        uint minpx = camera->width - raycaster->minPixelX;
        uint minpy = camera->height - raycaster->minPixelY;

        uint maxpx = camera->width - raycaster->maxPixelX;
        uint maxpy = camera->height - raycaster->maxPixelY;

        uint4 a = make_uint4(maxpx, maxpy, minpx, minpy);
        minpx = a.x;
        minpy = a.y;
        maxpx = a.z;
        maxpy = a.w;

//        if (x > minpx - 5 && x < minpx + 5 && y > minpy - 5 && y < minpy + 5) {
//            surf2Dwrite<uchar4>(make_uchar4(255, 255, 0, 255), raycaster->surface, x * sizeof(uchar4), y);
//            return;
//        }
//
//        if (x > maxpx - 5 && x < maxpx + 5 && y > maxpy - 5 && y < maxpy + 5) {
//            surf2Dwrite<uchar4>(make_uchar4(0, 255, 255, 255), raycaster->surface, x * sizeof(uchar4), y);
//            return;
//        }

        if (x >= minpx && x <= maxpx && y >= minpy && y <= maxpy) {
            Ray ray = SingleRayCaster::GetRay(vec2(x, y), camera);
            bool res = BBoxTminTmax(ray.origin, ray.dir, volume->bboxMin, volume->bboxMax, &ray.tmin, &ray.tmax);
            if (!res) {
                uchar4 element = make_uchar4(0, 0, 0, 0);
                surf2Dwrite<uchar4>(element, raycaster->surface, x * sizeof(uchar4), y);
                return;
            }

            /** Call forward. */
            vec4 result = forward(ray, volume) * 255.0f;
            uchar4 element = make_uchar4(result.x, result.y, result.z, 255.0f);
            surf2Dwrite<uchar4>(element, raycaster->surface, (x) * sizeof(uchar4), y);
        } else {
            uchar4 element = make_uchar4(0, 0, 0, 0);
            surf2Dwrite<uchar4>(element, raycaster->surface, x * sizeof(uchar4), y);
        }

    } else {
        Ray ray = SingleRayCaster::GetRay(vec2(x, y), camera);
        /** Call forward. */
        vec4 result = forward(ray, volume) * 255.0f;
        uchar4 element = make_uchar4(result.x, result.y, result.z, result.w);
        surf2Dwrite<uchar4>(element, raycaster->surface, (x) * sizeof(uchar4), y);
    }
}

extern "C" void volume_rendering_wrapper(GPUData<RayCasterDescriptor> &raycaster, GPUData<CameraDescriptor> &camera, GPUData<VolumeDescriptor> &volume) {
    /** Max 1024 per block. As each pixel is independent, may be useful to search for optimal size. */
    dim3 threadsPerBlock(16, 16);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 numBlocks(
            (camera.Host()->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (camera.Host()->height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    /** Call the main volumeRendering kernel. **/
    volumeRenderingUI8<<<numBlocks, threadsPerBlock>>>(raycaster.Device(), camera.Device(), volume.Device());

    /** Get last error after rendering. */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}


__global__ void batched_forward(VolumeDescriptor *volume, BatchItemDescriptor *item) {
    /** Pixel coords. */
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= item->cam->width || y >= item->cam->height) return;


    uchar4 ground_truth = make_uchar4(
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x)],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 1],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 2],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 3]
    );
    vec3 gt_color = UCHAR4_TO_VEC3(ground_truth) / 255.0f;
    auto alpha_gt = __uint2float_rn(ground_truth.w);

    Ray ray = SingleRayCaster::GetRay(ivec2(x, y), item->cam);
    bool bboxres = BBoxTminTmax(ray.origin, ray.dir, volume->bboxMin, volume->bboxMax, &ray.tmin, &ray.tmax);
//    if(!bboxres){
//        if (item->debugRender) {
//            uchar4 element = make_uchar4(255, 0, 0, 255);
//            surf2Dwrite<uchar4>(element, item->debugSurface, (x) * sizeof(uchar4), y);
//        }
//        return;
//    }

//    ray.tmin = item->range->data[LINEAR_IMG_INDEX(x, y, item->range->dim.y)].x;
//    ray.tmax = item->range->data[LINEAR_IMG_INDEX(x, y, item->range->dim.y)].y;

    ray.tmin = clamp(ray.tmin, 0.0f, INFINITY);
    ray.tmax = clamp(ray.tmax, 0.0f, INFINITY);

    /** Run forward function. */
    vec4 res = forward(ray, volume);
    item->cpred[LINEAR_IMG_INDEX(x, y, item->res.y)] = res;

    /** Store loss. */
    float epsilon = 0.001f;
    vec3 pred_color = vec3(res);
    vec3 loss = (gt_color - pred_color) / ((pred_color + epsilon) * (pred_color + epsilon));
    auto alpha_loss = (alpha_gt - res.w) / ((res.w + epsilon) * (res.w + epsilon));

    item->loss[LINEAR_IMG_INDEX(x, y, item->res.y)] = vec4(loss, alpha_loss);

    if (item->debugRender) {
        loss *= 255.0f;
        loss = clamp(loss, vec3(0.0, 0.0, 0.0), vec3(255.0, 255.0, 255.0));

        uchar4 element = VEC3_255_TO_UCHAR4((res * 255.0f));
        if(alpha_gt == 0.0f) {
            element.x = 255;
//            element.w = 255;

        }
//        uchar4 element = VEC3_255_TO_UCHAR4((res * 255.0f));
        surf2Dwrite<uchar4>(element, item->debugSurface, (x) * sizeof(uchar4), y);
    }

}

__global__ void batched_backward(VolumeDescriptor *volume, BatchItemDescriptor *item, AdamOptimizerDescriptor* adam) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= item->cam->width || y >= item->cam->height) return;

    Ray ray = SingleRayCaster::GetRay(ivec2(x, y), item->cam);
    bool bboxres = BBoxTminTmax(ray.origin, ray.dir, volume->bboxMin, volume->bboxMax, &ray.tmin, &ray.tmax);
    if(!bboxres) return;

    ray.tmin = clamp(ray.tmin, 0.0f, INFINITY);
    ray.tmax = clamp(ray.tmax, 0.0f, INFINITY);

    uchar4 ground_truth = make_uchar4(
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x)],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 1],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 2],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 3]
    );
    vec3 cgt = UCHAR4_TO_VEC3(ground_truth) / 255.0f;
    auto alpha_gt = __uint2float_rn(ground_truth.w);

    float epsilon = 0.001f;

    auto loss = item->loss[LINEAR_IMG_INDEX(x, y, item->res.y)];
    auto cpred = item->cpred[LINEAR_IMG_INDEX(x, y, item->res.y)];
    auto colorPred = vec3(cpred);
    auto dLdC = (2.0f * (colorPred - cgt)) / ((colorPred + vec3(epsilon)) * (colorPred + vec3(epsilon)));
    auto dLdalpha = (2.0f * (cpred.w - alpha_gt)) / ((cpred.w + vec3(epsilon)) * (cpred.w + vec3(epsilon)));

    dLdC = clamp(dLdC, -10.0f, 10.0f);

    /** Partial transmittance. */
    float Tpartial = 1.0f;
    /** Partial color. */
    vec3 Cpartial = vec3(0.0f, 0.0f, 0.0f);

    float step = 0.06f;

    /** The ray's min must be strictly smaller than max. */
    if (ray.tmin < ray.tmax) {

        /** Travel through the ray from it's min to max. */
        for (float t = ray.tmin; t < ray.tmax; t += step) {
            vec3 pos = ray.origin + t * ray.dir;

            if (IsPointInBBox(pos, volume)) {
                vec4 data = ReadVolume(pos, volume);
                vec3 color = vec3(data.r, data.g, data.b);
                float alpha = data.a;

//                Cpartial += Tpartial * alpha * color;
                Cpartial += Tpartial * (1 - exp(-alpha)) * color;

                /** Compute full loss */
                auto dLo_dCi = Tpartial * ( 1 - exp(-alpha));
                auto color_grad = dLdC * dLo_dCi;

                auto dCdAlpha = Tpartial * color * exp(-alpha) - (colorPred - color);

                auto alpha_grad = dot(dLdalpha, dCdAlpha);

                WriteVolumeTRI(pos, adam->grads, vec4(color_grad, alpha_grad));

//                Tpartial *= (1.0f - alpha);
                Tpartial *= exp(-alpha);

                if (Tpartial < 0.001f) {
                    Tpartial = 0.0f;
                    break;
                }
            }
        }
    }


}


extern "C" void batched_backward_wrapper(GPUData<BatchItemDescriptor>& item, GPUData<VolumeDescriptor>& volume, GPUData<AdamOptimizerDescriptor>& adam) {
    dim3 threads(16, 16);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 blocks((item.Host()->res.x + threads.x - 1) / threads.x,
                (item.Host()->res.y + threads.y - 1) / threads.y);

    batched_backward<<<blocks, threads>>>(volume.Device(), item.Device(), adam.Device());
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(batched_backward_wrapper) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}

extern "C" void batched_forward_wrapper(GPUData<BatchItemDescriptor> &item, GPUData<VolumeDescriptor> &volume) {
    dim3 threads(16, 16);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 blocks((item.Host()->res.x + threads.x - 1) / threads.x,
                (item.Host()->res.y + threads.y - 1) / threads.y);

    batched_forward<<<blocks, threads>>>(volume.Device(), item.Device());
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(batched_forward_wrapper) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}