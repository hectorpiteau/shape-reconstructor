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

__device__ bool IsPointInBBox(const vec3 &point, SparseVolumeDescriptor *volume) {
    if (all(lessThan(point, volume->bboxMax)) && all(greaterThan(point, volume->bboxMin)))
        return true;
    else
        return false;
}

__device__ bool IsPointInBBox(const vec3 &point, DenseVolumeDescriptor *volume) {
    if (all(lessThan(point, volume->bboxMax)) && all(greaterThan(point, volume->bboxMin)))
        return true;
    else
        return false;
}

__device__ inline float adaptive_step(vec4 &previous_data) {
    return clamp(1.0f / (5.0f * previous_data.w + 0.5f), 0.01f, 10.0f);
}

__device__ vec4 forward_sparse(Ray &ray, SparseVolumeDescriptor *volume, OneRayDebugInfoDescriptor *debugRay = NULL) {
    /** Partial transmittance. */
    float Tpartial = 1.0f;
    /** Partial color. */
    vec3 Cpartial = vec3(0.0f, 0.0f, 0.0f);
    float pstep = (volume->worldSize.x / (float) volume->res.x);
    float mstep = 0.5f;
    float step = pstep * mstep;

    int cpt = 0;
    vec4 data = {};

    size_t indices[8] = {};


    /** The ray's min must be strictly smaller than max. */
    if (ray.tmin < ray.tmax) {

        /** Travel through the ray from it's min to max. */
        for (float t = ray.tmin; t < ray.tmax; t += step) {
            vec3 pos = ray.origin + t * ray.dir;

            if (IsPointInBBox(pos, volume)) {
                data = ReadVolume(pos, volume, indices);

                if (debugRay != NULL && debugRay->active) {
                    debugRay->pointsWorldCoords[cpt] = pos;
                    debugRay->pointsSamples[cpt] = data;
                    debugRay->points = cpt;
                    cpt += 1;
                }

                vec3 color = vec3(data.r, data.g, data.b);
                float alpha = data.a;

                alpha = clamp(alpha, 0.0f, 0.9999f);

//                mstep = 0.6f + (0.6f - 0.1f) * alpha;
//                mstep = clamp(mstep, 0.1f, 0.6f);
//                step = pstep * mstep;

                color = clamp(color, vec3(0.0), vec3(1.0));

                Cpartial += Tpartial * alpha * color;
                Tpartial *= (1.0f - alpha);

                if (Tpartial < 0.001f) {
                    Tpartial = 0.0f;
                    break;
                }
            }
        }
    }
    return {Cpartial, Tpartial};
}

__device__ vec4 forward(Ray &ray, DenseVolumeDescriptor *volume) {
    /** Partial transmittance. */
    float Tpartial = 1.0f;
    /** Partial color. */
    vec3 Cpartial = vec3(0.0f, 0.0f, 0.0f);

    float step = (volume->worldSize.x / (float) volume->res.x) * 0.5f;

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

                Cpartial += Tpartial * alpha * color;
                Tpartial *= (1.0f - alpha);

                if (Tpartial < 0.001f) {
                    Tpartial = 0.0f;
                    break;
                }
            }
        }
    }
    return {Cpartial, Tpartial};
}


__global__ void
volumeRenderingUI8(RayCasterDescriptor *raycaster, CameraDescriptor *camera, DenseVolumeDescriptor *volume) {
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

__global__ void
sparseVolumeRenderingUI8(RayCasterDescriptor *raycaster, CameraDescriptor *camera, SparseVolumeDescriptor *volume,
                         OneRayDebugInfoDescriptor *debugRay) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= camera->width || y >= camera->height) return;

    bool debugRayValid = false;
    if (x == debugRay->pixelCoords.x && (1080 - y) == debugRay->pixelCoords.y) {
        debugRayValid = true;
    }

//    if (!raycaster->renderAllPixels) {
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
        vec4 result = forward_sparse(ray, volume, debugRayValid ? debugRay : NULL);
        result.w = 1.0f - result.w;
        result *= 255.0f;
        uchar4 element = VEC4_TO_UCHAR4(result);
        surf2Dwrite<uchar4>(element, raycaster->surface, (x) * sizeof(uchar4), y);
    } else {
        uchar4 element = make_uchar4(0, 0, 0, 0);
        surf2Dwrite<uchar4>(element, raycaster->surface, x * sizeof(uchar4), y);
    }
//    } else {
//        Ray ray = SingleRayCaster::GetRay(vec2(x, y), camera);
//        /** Call forward. */
//        vec4 result = forward_sparse(ray, volume) * 255.0f;
//        uchar4 element = VEC4_TO_UCHAR4(result);
//        surf2Dwrite<uchar4>(element, raycaster->surface, (x) * sizeof(uchar4), y);
//    }
}

//extern "C" void volume_rendering_wrapper(GPUData<RayCasterDescriptor> &raycaster, GPUData<CameraDescriptor> &camera, GPUData<DenseVolumeDescriptor> &volume) {
//    /** Max 1024 per block. As each pixel is independent, may be useful to search for optimal size. */
//    dim3 threadsPerBlock(16, 16);
//    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
//    dim3 numBlocks(
//            (camera.Host()->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
//            (camera.Host()->height + threadsPerBlock.y - 1) / threadsPerBlock.y);
//
//    /** Call the main volumeRendering kernel. **/
//    volumeRenderingUI8<<<numBlocks, threadsPerBlock>>>(raycaster.Device(), camera.Device(), volume.Device());
//
//    /** Get last error after rendering. */
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
//    }
//
//    cudaDeviceSynchronize();
//}

extern "C" void
sparse_volume_rendering_wrapper(GPUData<RayCasterDescriptor> &raycaster, GPUData<CameraDescriptor> &camera,
                                GPUData<SparseVolumeDescriptor> *volume, GPUData<OneRayDebugInfoDescriptor> *debugRay) {
    /** Max 1024 per block. As each pixel is independent, may be useful to search for optimal size. */
    dim3 threadsPerBlock(16, 16);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 numBlocks(
            (camera.Host()->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (camera.Host()->height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    /** Call the main volumeRendering kernel. **/
    sparseVolumeRenderingUI8<<<numBlocks, threadsPerBlock>>>(raycaster.Device(), camera.Device(), volume->Device(),
                                                             debugRay->Device());

    /** Get last error after rendering. */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}

__device__ void
forward_one_ray(unsigned int x, unsigned int y, Ray &ray, DenseVolumeDescriptor *volume, BatchItemDescriptor *item,
                uchar4 ground_truth, SuperResolutionDescriptor *superRes, int superResIndex = 0) {

    vec3 gt_color = UCHAR4_TO_VEC3(ground_truth) / vec3(255.0f);
    auto alpha_gt = __uint2float_rn(ground_truth.w);

    bool bboxres = BBoxTminTmax(ray.origin, ray.dir, volume->bboxMin, volume->bboxMax, &ray.tmin, &ray.tmax);

    ray.tmin = clamp(ray.tmin, 0.0f, INFINITY);
    ray.tmax = clamp(ray.tmax, 0.0f, INFINITY);

    /** Run forward function. */
    vec4 res = forward(ray, volume);
    res = clamp(res, vec4(0.0), vec4(1.0));
    item->cpred[LINEAR_IMG_INDEX(x, y, item->res, superResIndex)] = res;

    /** Store loss. */
    float epsilon = 1.0E-8f;
    vec3 pred_color = vec3(res);
    vec3 loss = (gt_color - pred_color) / ((pred_color + epsilon) * (pred_color + epsilon));
    auto alpha_loss = (alpha_gt - res.w) / ((res.w + epsilon) * (res.w + epsilon));

    item->loss[LINEAR_IMG_INDEX(x, y, item->res, superResIndex)] = vec4(loss, alpha_loss);

    /** PSNR */
    auto psnr_diff = pow(gt_color - pred_color, vec3(2));
    atomicAdd(&item->psnr.x, psnr_diff.x);
    atomicAdd(&item->psnr.y, psnr_diff.y);
    atomicAdd(&item->psnr.z, psnr_diff.z);

    if (item->debugRender) {
        uchar4 element;
        switch (item->mode) {
            case RenderMode::COLOR_LOSS:
                loss *= 255.0f;
                loss = clamp(loss, vec3(0.0, 0.0, 0.0), vec3(255.0, 255.0, 255.0));
                element = VEC3_255_TO_UCHAR4((loss));
                break;
            case RenderMode::ALPHA_LOSS:
                alpha_loss = clamp(alpha_loss, 0.0f, 255.0f);
                element = VEC3_255_TO_UCHAR4((vec3(alpha_loss, alpha_loss, alpha_loss)));
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

__device__ void sparse_forward_one_ray(unsigned int x, unsigned int y, Ray &ray, SparseVolumeDescriptor *volume,
                                       BatchItemDescriptor *item, uchar4 &ground_truth,
                                       SuperResolutionDescriptor *superRes, unsigned int superRedIndex) {
    vec3 gt_color = UCHAR4_TO_VEC3(ground_truth) / 255.0f;
    auto alpha_gt = __uint2float_rn(ground_truth.w);

    bool bboxres = BBoxTminTmax(ray.origin, ray.dir, volume->bboxMin, volume->bboxMax, &ray.tmin, &ray.tmax);

    ray.tmin = clamp(ray.tmin, 0.0f, INFINITY);
    ray.tmax = clamp(ray.tmax, 0.0f, INFINITY);

    /** Run forward function. */
    vec4 res = forward_sparse(ray, volume);
    item->cpred[LINEAR_IMG_INDEX(x, y, item->res, superRedIndex)] = res;

    /** Store loss. */
    float epsilon = 1.0E-8f;
    vec3 pred_color = vec3(res);
//    vec3 loss = (gt_color - pred_color) / ((pred_color + epsilon) * (pred_color + epsilon));
//    auto alpha_loss = (alpha_gt - res.w) / ((res.w + epsilon) * (res.w + epsilon));

//    item->loss[LINEAR_IMG_INDEX(x, y, item->res, 0)] = vec4(loss, alpha_loss);

    auto psnr_diff = pow(gt_color - pred_color, vec3(2));
    atomicAdd(&item->psnr.x, psnr_diff.x);
    atomicAdd(&item->psnr.y, psnr_diff.y);
    atomicAdd(&item->psnr.z, psnr_diff.z);

    if (item->debugRender) {
        uchar4 element;
        switch (item->mode) {
            case RenderMode::COLOR_LOSS:
//                loss *= 255.0f;
//                loss = clamp(loss, vec3(0.0, 0.0, 0.0), vec3(255.0, 255.0, 255.0));
//                element = VEC3_255_TO_UCHAR4((loss));
                break;
            case RenderMode::ALPHA_LOSS:
//                alpha_loss = clamp(alpha_loss, 0.0f, 255.0f);
//                element = VEC3_255_TO_UCHAR4((vec3(alpha_loss, alpha_loss, alpha_loss)));
                break;
            case RenderMode::PREDICTED_COLOR:
                pred_color *= 255.0f;
                pred_color = clamp(pred_color, 0.0f, 255.0f);
//                element = VEC3_255_TO_UCHAR4(pred_color);
                break;
            case RenderMode::GROUND_TRUTH:
            default:
                element = ground_truth;
                break;

        }

        surf2Dwrite<uchar4>(element, item->debugSurface, (x) * sizeof(uchar4), y);
    }
}

__global__ void
batched_forward_sparse(SparseVolumeDescriptor *volume, BatchItemDescriptor *item, SuperResolutionDescriptor *superRes) {
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


    if (superRes->active) {
        for (int i = 0; i < superRes->raysAmount; ++i) {
            Ray ray = SingleRayCaster::GetRay(vec2(((float)x) + superRes->shifts[i].x, ((float)y)) + superRes->shifts[i].y, item->cam);
            sparse_forward_one_ray(
                    x,
                    y,
                    ray,
                    volume,
                    item,
                    ground_truth,
                    superRes,
                    i);
        }
    } else {
        Ray ray = SingleRayCaster::GetRay(ivec2(x, y), item->cam);
        sparse_forward_one_ray(x, y, ray, volume, item, ground_truth, superRes, 0);
    }
}


__global__ void
batched_forward(DenseVolumeDescriptor *volume, BatchItemDescriptor *item, SuperResolutionDescriptor *superRes) {
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

    if (superRes->active) {
        for (int i = 0; i < superRes->raysAmount; ++i) {
            Ray ray = SingleRayCaster::GetRay(vec2(x, y), item->cam);
            forward_one_ray(
                    x + superRes->shifts[i].x,
                    y + superRes->shifts[i].y,
                    ray, volume,
                    item,
                    ground_truth,
                    superRes,
                    i);
        }
    } else {
        Ray ray = SingleRayCaster::GetRay(ivec2(x, y), item->cam);
        forward_one_ray(x, y, ray, volume, item, ground_truth, superRes);
    }
}

__device__ void backward_one_ray(unsigned int x, unsigned int y,
                                 uchar4 ground_truth,
                                 Ray &ray,
                                 BatchItemDescriptor *item,
                                 DenseVolumeDescriptor *volume = NULL,
                                 AdamOptimizerDescriptor *adam = NULL,
                                 SuperResolutionDescriptor *super_res = NULL,
                                 SparseVolumeDescriptor *s_volume = NULL,
                                 SparseAdamOptimizerDescriptor *s_adam = NULL,
                                 bool use_sparse = false,
                                 bool use_super_res = false,
                                 unsigned int srIndex = 0) {
    bool bboxres = false;
    if (use_sparse)
        bboxres = BBoxTminTmax(ray.origin, ray.dir, s_volume->bboxMin, s_volume->bboxMax, &ray.tmin, &ray.tmax);
    else
        bboxres = BBoxTminTmax(ray.origin, ray.dir, volume->bboxMin, volume->bboxMax, &ray.tmin, &ray.tmax);

    if (!bboxres) return;

    ray.tmin = clamp(ray.tmin, 0.0f, INFINITY);
    ray.tmax = clamp(ray.tmax, 0.0f, INFINITY);

    vec3 cgt = UCHAR4_TO_VEC3(ground_truth) / 255.0f;
    auto alpha_gt = 1.0f - (__uint2float_rn(ground_truth.w) / 255.0f);

    float epsilon = 1.0E-8f;

//    auto loss = item->loss[LINEAR_IMG_INDEX(x, y, item->res.y)];
    auto cpred = item->cpred[LINEAR_IMG_INDEX(x, y, item->res, srIndex)];
    auto colorPred = vec3(cpred);
//    auto dLdC = (2.0f * (colorPred - cgt)) / ((colorPred + vec3(epsilon)) * (colorPred + vec3(epsilon)));
    auto dLdC = (2.0f * (colorPred - cgt));

//    auto dLdalpha = (2.0f * (cpred.w - alpha_gt)) / ((cpred.w + epsilon) * (cpred.w + epsilon));

    dLdC = clamp(dLdC, -100.0f, 100.0f);

    auto Tinf = cpred.w;

    /** Partial transmittance. */
    float Tpartial = 1.0f;
    /** Partial color. */
    vec3 Cpartial = vec3(0.0f, 0.0f, 0.0f);

    float step = 0.1f;
    if (use_sparse)
        step = (s_volume->worldSize.x / (float) s_volume->res.x) * 0.25f;
    else
        step = (volume->worldSize.x / (float) volume->res.x) * 0.5f;

    size_t indices[8] = {};

    /** The ray's min must be strictly smaller than max. */
    if (ray.tmin < ray.tmax) {

        /** Travel through the ray from it's min to max. */
        for (float t = ray.tmin; t < ray.tmax; t += step) {
            vec3 pos = ray.origin + t * ray.dir;
            bool isInBox = use_sparse ? IsPointInBBox(pos, s_volume) : IsPointInBBox(pos, volume);
            if (isInBox) {
                vec4 data{};
                if (use_sparse) {
                    data = ReadVolume(pos, s_volume, indices);
                } else {
                    data = ReadVolume(pos, volume, indices);
                }

                vec3 color = vec3(data.r, data.g, data.b);
                float alpha = data.a;
                alpha = clamp(alpha, 0.0f, 0.99f);

//                Cpartial += Tpartial * (1 - exp(-alpha)) * color;
                Cpartial += Tpartial * alpha * color;

                /** Compute full loss */
                auto color_0_w = use_sparse ? s_adam->color_0_w : adam->color_0_w;
                auto alpha_0_w = use_sparse ? s_adam->alpha_0_w : adam->alpha_0_w;
                auto alpha_reg_0_w = use_sparse ? s_adam->alpha_reg_0_w : adam->alpha_reg_0_w;

                auto dLo_dCi = Tpartial * (alpha);
                auto color_grad = color_0_w * dLdC * dLo_dCi;

//                vec3 posp1 = (ray.origin + (t+1) * ray.dir);
//                auto c_k1 = ReadVolume(posp1, volume);  //TEST WITH COLOR_k+1
                auto dCdAlpha = Tpartial * color - (colorPred - color) / (1.0f - alpha);


//                auto alpha_reg_i = 2.0f * (-alpha * (1.0f - cpred.w)) - 2.0f * alpha_gt * ( -alpha * (1.0f - cpred.w));
                auto alpha_reg_i = 2.0f * (Tinf - alpha_gt) * Tinf / (1.0f - alpha);

                auto alpha_grad = alpha_0_w * dot(dLdC, dCdAlpha) + alpha_reg_0_w * alpha_reg_i;

                if (use_super_res && super_res->active) {
                    auto gaussian_weight = GeneralGaussian2D(super_res->shifts[srIndex].x, super_res->shifts[srIndex].y);
//                    auto gaussian_weight = 0.25f;
                    alpha_grad = alpha_grad * gaussian_weight;
                    color_grad = color_grad * gaussian_weight;
                }

                if (use_sparse) {
                    //color_grad, alpha_grad
                    WriteVolumeTRI(pos, s_adam->grads, vec4(color_grad, alpha_grad), indices, s_adam);
                } else {
                    WriteVolumeTRI(pos, adam->grads, vec4(color_grad, alpha_grad), indices, adam);
                }

                Tpartial *= (1.0f - alpha);

                if (Tpartial < 0.001f) {
                    Tpartial = 0.0f;
                    break;
                }
            }
        }
    }
}

__global__ void
BatchedBackwardSparse(SparseVolumeDescriptor *volume, BatchItemDescriptor *item, SparseAdamOptimizerDescriptor *adam,
                      SuperResolutionDescriptor *superRes) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= item->cam->width || y >= item->cam->height) return;

    uchar4 gt = make_uchar4(
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x)],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 1],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 2],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 3]
    );

    if (superRes->active) {
        // superRes->raysAmount
        for (int i = 0; i < 4; ++i) {
            // superRes->shifts[i].x  superRes->shifts[i].y
            Ray ray = SingleRayCaster::GetRay(vec2(((float)x) + superRes->shifts[i].x, ((float)y)) + superRes->shifts[i].y, item->cam);

            backward_one_ray(x, y, gt, ray, item, NULL, NULL, superRes, volume, adam, true, true, i);
        }
    } else {
        Ray ray = SingleRayCaster::GetRay(ivec2(x, y), item->cam);
//        bool bboxres = BBoxTminTmax(ray.origin, ray.dir, volume->bboxMin, volume->bboxMax, &ray.tmin, &ray.tmax);
//        if (!bboxres) return;
//        ray.tmin = clamp(ray.tmin, 0.0f, INFINITY);
//        ray.tmax = clamp(ray.tmax, 0.0f, INFINITY);

        backward_one_ray(x, y, gt, ray, item, NULL, NULL, superRes, volume, adam, true, false, 0);
    }
}


__global__ void BatchedBackward(DenseVolumeDescriptor *volume, BatchItemDescriptor *item, AdamOptimizerDescriptor *adam,
                                SuperResolutionDescriptor *superRes) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= item->cam->width || y >= item->cam->height) return;

    uchar4 gt = make_uchar4(
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x)],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 1],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 2],
            item->img->data[STBI_IMG_INDEX(x, y, item->img->res.x) + 3]
    );

    if (superRes->active) {
        for (int i = 0; i < superRes->raysAmount; ++i) {
            Ray ray = SingleRayCaster::GetRay(vec2(x + superRes->shifts[i].x, y + superRes->shifts[i].y), item->cam);
            backward_one_ray(x, y, gt, ray, item, volume, adam, superRes, NULL, NULL, false, true, i);
        }
    } else {
        Ray ray = SingleRayCaster::GetRay(ivec2(x, y), item->cam);
        backward_one_ray(x, y, gt, ray, item, volume, adam, superRes, NULL, NULL, false, false, 0);
    }
}


extern "C" void batched_backward_wrapper(GPUData<BatchItemDescriptor> &item, GPUData<DenseVolumeDescriptor> &volume,
                                         GPUData<AdamOptimizerDescriptor> &adam,
                                         GPUData<SuperResolutionDescriptor> &superRes) {
    dim3 threads(16, 16);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 blocks((item.Host()->res.x + threads.x - 1) / threads.x,
                (item.Host()->res.y + threads.y - 1) / threads.y);

    BatchedBackward<<<blocks, threads>>>(volume.Device(), item.Device(), adam.Device(), superRes.Device());
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(batched_backward_wrapper) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}

extern "C" void
batched_backward_sparse_wrapper(GPUData<BatchItemDescriptor> *item, GPUData<SparseVolumeDescriptor> *volume,
                                GPUData<SparseAdamOptimizerDescriptor> *adam,
                                GPUData<SuperResolutionDescriptor> *superRes) {
    dim3 threads(8, 8);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 blocks((item->Host()->res.x + threads.x - 1) / threads.x,
                (item->Host()->res.y + threads.y - 1) / threads.y);

    BatchedBackwardSparse<<<blocks, threads>>>(volume->Device(), item->Device(), adam->Device(), superRes->Device());
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(batched_backward_sparse_wrapper) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}

extern "C" void
batched_forward_sparse_wrapper(GPUData<BatchItemDescriptor> &item, GPUData<SparseVolumeDescriptor> *volume,
                               GPUData<SuperResolutionDescriptor> *superRes) {
    dim3 threads(8, 8);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 blocks((item.Host()->res.x + threads.x - 1) / threads.x,
                (item.Host()->res.y + threads.y - 1) / threads.y);

    batched_forward_sparse<<<blocks, threads>>>(volume->Device(), item.Device(), superRes->Device());
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(batched_forward_wrapper) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}

extern "C" void batched_forward_wrapper(GPUData<BatchItemDescriptor> &item, GPUData<DenseVolumeDescriptor> &volume,
                                        GPUData<SuperResolutionDescriptor> &superRes) {
    dim3 threads(16, 16);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 blocks((item.Host()->res.x + threads.x - 1) / threads.x,
                (item.Host()->res.y + threads.y - 1) / threads.y);

    batched_forward<<<blocks, threads>>>(volume.Device(), item.Device(), superRes.Device());
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(batched_forward_wrapper) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}

__global__ void sparse_volume_gradients(SparseVolumeDescriptor *volume, SparseAdamOptimizerDescriptor *adam) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > adam->res.x || y > adam->res.y || z > adam->res.z) return;

    auto indexes = SparseVolumeGetDataIndex(ivec3(x, y, z), adam->target);

    if (indexes.data_index == INF) return;

    /** Retrieve voxel's data. */
    cell d = volume->data[indexes.data_index];
    vec4 center_data = vec4(d.data.x, d.data.y, d.data.z, d.data.w);

    /** Create loss variables to store partial gradients. */
    vec4 tvl2_full_loss = vec4(0.0f);

    /** Check if not on the min border x. */
    int tmp_x = ((int) x) - 1;
    if (tmp_x > 0) {
        auto ind = SparseVolumeGetDataIndex(ivec3(tmp_x, y, z), volume);
        if (ind.data_index != INF) {
            auto tmp = float4ToVec4(volume->data[ind.data_index].data);
            tvl2_full_loss += 2.0f * (center_data - tmp);
        }
    }
    tmp_x = (int) (x) + 1;
    if (tmp_x < adam->res.x) {
        auto ind = SparseVolumeGetDataIndex(ivec3(tmp_x, y, z), volume);
        if (ind.data_index != INF) {
            auto tmp = float4ToVec4(volume->data[ind.data_index].data);
            tvl2_full_loss += 2.0f * (center_data - tmp);
        }
    }
    int tmp_y = (int) (y) - 1;
    if (tmp_y > 0) {
        auto ind = SparseVolumeGetDataIndex(ivec3(x, tmp_y, z), volume);
        if (ind.data_index != INF) {
            auto tmp = float4ToVec4(volume->data[ind.data_index].data);
            tvl2_full_loss += 2.0f * (center_data - tmp);
        }
    }
    tmp_y = (int) (y) + 1;
    if (tmp_y < volume->res.y) {
        auto ind = SparseVolumeGetDataIndex(ivec3(x, tmp_y, z), volume);
        if (ind.data_index != INF) {
            auto tmp = float4ToVec4(volume->data[ind.data_index].data);
            tvl2_full_loss += 2.0f * (center_data - tmp);
        }
    }
    int tmp_z = (int) (z) - 1;
    if (tmp_z > 0) {
        auto ind = SparseVolumeGetDataIndex(ivec3(x, y, tmp_z), volume);
        if (ind.data_index != INF) {
            auto tmp = float4ToVec4(volume->data[ind.data_index].data);
            tvl2_full_loss += 2.0f * (center_data - tmp);
        }
    }
    tmp_z = (int) (z) + 1;
    if (tmp_z < volume->res.z) {
        auto ind = SparseVolumeGetDataIndex(ivec3(x, y, tmp_z), volume);
        if (ind.data_index != INF) {
            auto tmp = float4ToVec4(volume->data[ind.data_index].data);
            tvl2_full_loss += 2.0f * (center_data - tmp);
        }
    }

    /** Write gradients. */
    tvl2_full_loss = tvl2_full_loss * adam->tvl2_0_w;

    atomicAdd(&(adam->grads->data[indexes.data_index].data.x), tvl2_full_loss.x);
    atomicAdd(&(adam->grads->data[indexes.data_index].data.y), tvl2_full_loss.y);
    atomicAdd(&(adam->grads->data[indexes.data_index].data.z), tvl2_full_loss.z);
    atomicAdd(&(adam->grads->data[indexes.data_index].data.w), tvl2_full_loss.w);
}

__global__ void volume_gradients(DenseVolumeDescriptor *volume, AdamOptimizerDescriptor *adam) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > adam->res.x || y > adam->res.y || z > adam->res.z) return;

    auto index = VOLUME_INDEX(x, y, z, volume->res);

    /** Retrieve voxel's data. */
    cell d = volume->data[index];
    vec3 color = vec3(d.data.x, d.data.y, d.data.z);
    float alpha = d.data.w;

    /** Create loss variables to store partial gradients. */
    vec3 tvl2_loss = vec3(0.0f);
    float tvl2_loss_alpha = 0.0f;

    /** Check if not on the min border x. */
    int tmp_x = x - 1;
    if (tmp_x > 0) {
        auto d_x_0 = float4ToVec4(volume->data[VOLUME_INDEX(tmp_x, y, z, volume->res)].data);
        tvl2_loss += 2.0f * (color - vec3(d_x_0));
        tvl2_loss_alpha += 2.0f * (alpha - d_x_0.w);
    }
    tmp_x = x + 1;
    if (tmp_x < volume->res.x) {
        auto d_x_1 = float4ToVec4(volume->data[VOLUME_INDEX(tmp_x, y, z, volume->res)].data);
        tvl2_loss += 2.0f * (color - vec3(d_x_1));
        tvl2_loss_alpha += 2.0f * (alpha - d_x_1.w);
    }
    int tmp_y = y - 1;
    if (tmp_y > 0) {
        auto d_y_0 = float4ToVec4(volume->data[VOLUME_INDEX(x, tmp_y, z, volume->res)].data);
        tvl2_loss += 2.0f * (color - vec3(d_y_0));
        tvl2_loss_alpha += 2.0f * (alpha - d_y_0.w);
    }
    tmp_y = y + 1;
    if (tmp_y < volume->res.y) {
        auto d_y_1 = float4ToVec4(volume->data[VOLUME_INDEX(x, tmp_y, z, volume->res)].data);
        tvl2_loss += 2.0f * (color - vec3(d_y_1));
        tvl2_loss_alpha += 2.0f * (alpha - d_y_1.w);
    }
    int tmp_z = z - 1;
    if (tmp_z > 0) {
        auto d_z_0 = float4ToVec4(volume->data[VOLUME_INDEX(x, y, tmp_z, volume->res)].data);
        tvl2_loss += 2.0f * (color - vec3(d_z_0));
        tvl2_loss_alpha += 2.0f * (alpha - d_z_0.w);
    }
    tmp_z = z + 1;
    if (tmp_z < volume->res.z) {
        auto d_z_1 = float4ToVec4(volume->data[VOLUME_INDEX(x, y, tmp_z, volume->res)].data);
        tvl2_loss += 2.0f * (color - vec3(d_z_1));
        tvl2_loss_alpha += 2.0f * (alpha - d_z_1.w);
    }

    /** Write gradients. */
    tvl2_loss *= adam->tvl2_0_w;
    tvl2_loss_alpha *= adam->tvl2_0_w;
    auto tmp = adam->grads->data[VOLUME_INDEX(x, y, z, adam->res)].data;
    adam->grads->data[VOLUME_INDEX(x, y, z, adam->res)].data = make_float4(
            tmp.x + tvl2_loss.x,
            tmp.y + tvl2_loss.y,
            tmp.z + tvl2_loss.z,
            tmp.w + tvl2_loss_alpha);
}

extern "C" void
sparse_volume_backward(GPUData<SparseVolumeDescriptor> *volume, GPUData<SparseAdamOptimizerDescriptor> *adam) {
    dim3 threads(4, 4, 4);
    /** This create enough blocks to cover the whole volume, may contain threads that does not have pixel's assigned. */
    dim3 blocks((adam->Host()->res.x + threads.x - 1) / threads.x,
                (adam->Host()->res.y + threads.y - 1) / threads.y,
                (adam->Host()->res.z + threads.z - 1) / threads.z);

    sparse_volume_gradients<<<blocks, threads>>>(volume->Device(), adam->Device());
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(sparse_volume_backward) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}

extern "C" void volume_backward(GPUData<DenseVolumeDescriptor> *volume, GPUData<AdamOptimizerDescriptor> *adam) {
    dim3 threads(4, 4, 4);
    /** This create enough blocks to cover the whole volume, may contain threads that does not have pixel's assigned. */
    dim3 blocks((adam->Host()->res.x + threads.x - 1) / threads.x,
                (adam->Host()->res.y + threads.y - 1) / threads.y,
                (adam->Host()->res.z + threads.z - 1) / threads.z);

    volume_gradients<<<blocks, threads>>>(volume->Device(), adam->Device());
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(volume_backward) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}