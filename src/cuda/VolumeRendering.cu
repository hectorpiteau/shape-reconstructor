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

__device__ vec4 forward(Ray &ray, VolumeDescriptor *volume) //, float4* volume, const ivec3& resolution)
{
    /** Partial transmittance. */
    float Tpartial = 1.0f;
    /** Partial color. */
    vec3 Cpartial = vec3(0.0f, 0.0f, 0.0f);

    float step = (volume->worldSize.x / (float)volume->res.x) * 0.5f;

    size_t indices[8] = {};

    /** The ray's min must be strictly smaller than max. */
    if (ray.tmin < ray.tmax) {

        /** Travel through the ray from it's min to max. */
        for (float t = ray.tmin; t < ray.tmax; t += step) {
            vec3 pos = ray.origin + t * ray.dir;

            if (IsPointInBBox(pos, volume)) {
                vec4 data = ReadVolume(pos, volume, indices);
                vec3 color = vec3(data.r, data.g, data.b);
                float alpha = data.a;
                alpha = clamp(alpha, 0.0f, 0.99f);

//                Cpartial += Tpartial * (1 - exp(-alpha)) * color;
                Cpartial += Tpartial * alpha * color;
                Tpartial *= (1.0f - alpha);
//                Tpartial *= (1.0f / exp(alpha));

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

        if (x >= minpx && x <= maxpx && y >= minpy && y <= maxpy) {
            Ray ray = SingleRayCaster::GetRay(vec2(x, y), camera);
            bool res = BBoxTminTmax(ray.origin, ray.dir, volume->bboxMin, volume->bboxMax, &ray.tmin, &ray.tmax);
            if (!res) {
                uchar4 element = make_uchar4(0, 0, 0, 0);
                surf2Dwrite<uchar4>(element, raycaster->surface, x * sizeof(uchar4), y);
                return;
            }

            /** Call forward. */
            vec4 result = forward(ray, volume);
            result.w = 1.0f - result.w;
            result *= 255.0f;
            uchar4 element = make_uchar4(result.x, result.y, result.z, result.w);
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
        uchar4 element;
        switch(item->mode){
            case RenderMode::COLOR_LOSS:
                loss *= 255.0f;
                loss = clamp(loss, vec3(0.0, 0.0, 0.0), vec3(255.0, 255.0, 255.0));
                element = VEC3_255_TO_UCHAR4((loss));
                break;
            case RenderMode::ALPHA_LOSS:
                alpha_loss = clamp(alpha_loss, 0.0f, 255.0f);
                element = VEC3_255_TO_UCHAR4((vec3(alpha_loss,alpha_loss,alpha_loss)));
                break;
            case RenderMode::PREDICTED_COLOR:
                pred_color *= 255.0f;
                pred_color = clamp(pred_color, 0.0f, 255.0f);
                element = VEC3_255_TO_UCHAR4(pred_color);
                break;
            case RenderMode::GROUND_TRUTH:
            default:
                element = ground_truth;
                break;

        }

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
    auto alpha_gt = 1.0f - (__uint2float_rn(ground_truth.w) / 255.0f);

    float epsilon = 0.001f;

//    auto loss = item->loss[LINEAR_IMG_INDEX(x, y, item->res.y)];
    auto cpred = item->cpred[LINEAR_IMG_INDEX(x, y, item->res.y)];
    auto colorPred = vec3(cpred);
//    auto dLdC = (2.0f * (colorPred - cgt)) / ((colorPred + vec3(epsilon)) * (colorPred + vec3(epsilon)));
    auto dLdC = (2.0f * (colorPred - cgt));


//    auto dLdalpha = (2.0f * (cpred.w - alpha_gt)) / ((cpred.w + epsilon) * (cpred.w + epsilon));

    dLdC = clamp(dLdC, -10.0f, 10.0f);

    auto Tinf = cpred.w;

    /** Partial transmittance. */
    float Tpartial = 1.0f;
    /** Partial color. */
    vec3 Cpartial = vec3(0.0f, 0.0f, 0.0f);

    float step = (volume->worldSize.x / (float)volume->res.x) * 0.5f;

    size_t indices[8] = {};

    /** The ray's min must be strictly smaller than max. */
    if (ray.tmin < ray.tmax) {

        /** Travel through the ray from it's min to max. */
        for (float t = ray.tmin; t < ray.tmax; t += step) {
            vec3 pos = ray.origin + t * ray.dir;

            if (IsPointInBBox(pos, volume)) {
                vec4 data = ReadVolume(pos, volume, indices);
                vec3 color = vec3(data.r, data.g, data.b);
                float alpha = data.a;
                alpha = clamp(alpha, 0.0f, 0.99f);

//                Cpartial += Tpartial * (1 - exp(-alpha)) * color;
                Cpartial += Tpartial * alpha * color;

                /** Compute full loss */
                auto dLo_dCi = Tpartial * (alpha);
                auto color_grad = adam->color_0_w * dLdC * dLo_dCi;

//                vec3 posp1 = (ray.origin + (t+1) * ray.dir);
//                auto c_k1 = ReadVolume(posp1, volume);  //TEST WITH COLOR_k+1
                auto dCdAlpha = Tpartial * color - (colorPred - color) / (1.0f - alpha);


//                auto alpha_reg_i = 2.0f * (-alpha * (1.0f - cpred.w)) - 2.0f * alpha_gt * ( -alpha * (1.0f - cpred.w));
                auto alpha_reg_i = 2.0f * (Tinf - alpha_gt) * Tinf / (1.0f - alpha);

                auto alpha_grad = adam->alpha_0_w *  dot(dLdC, dCdAlpha) + adam->alpha_reg_0_w *  alpha_reg_i;


                WriteVolumeTRI(pos, adam->grads, vec4(color_grad, alpha_grad), indices);

//                Tpartial *= exp(-alpha);
                Tpartial *= (1.0f - alpha);

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