//
// Created by hpiteau on 09/06/23.
//

#ifndef DRTMCS_CONVOLUTIONS_CUH
#define DRTMCS_CONVOLUTIONS_CUH

#include <glm/glm.hpp>
#include "Common.cuh"
#include <math_constants.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif


/**
 * Perform a gaussian convolution on a source image and write the result in the target image.
 * Source and Target images have to be of the same size.
 *
 * BORDER : (default) CLAMP
 *
 * @param source : The descriptor for the first image where values will be taken from.
 * @param target : Where the function will write the result.
 * @param weights : Gaussian kernel to apply, struct containing weights.
 * @param center : The coordinates
 */
//CUDA_DEV inline void ImageGaussianFloat(ImageDescriptor *source, ImageDescriptor *target, GaussianWeightsDescriptor &weights,
//                                        const glm::ivec2 &center) {
//    if (weights.dim != 2) return;
//    unsigned short sizeDiv2 = floor((float) weights.size / 2.0f);
//    glm::vec4 result = {0.0, 0.0, 0.0, 0.0};
//    glm::vec4 tmp{};
//    for (unsigned short y = -sizeDiv2; y < sizeDiv2 + 1; y++) {
//        for (unsigned short x = -sizeDiv2; x < sizeDiv2 + 1; x++) {
//            surf2Dread((float4 * ) & tmp[0], source->surface, (center.x + x) * sizeof(float4), (center.y + y),
//                       cudaBoundaryModeClamp);
//            result += tmp * weights.weights[(y + sizeDiv2) + (x + sizeDiv2) * weights.size];
//        }
//    }
//    surf2Dwrite<float4>(result, target->surface, center.x * sizeof(float4), center.y);
//}

CUDA_DEV inline void
//VoxelGaussian(VolumeDescriptor *source, VolumeDescriptor *target, GaussianWeightsDescriptor &weights, const glm::ivec2 &center) {
//    if (weights.dim != 3) return;
//    unsigned short sizeDiv2 = floor((float) weights.size / 2.0f);
//    glm::vec4 result = {0.0, 0.0, 0.0, 0.0};
//    glm::vec4 tmp{};
//    for (unsigned short y = -sizeDiv2; y < sizeDiv2 + 1; y++) {
//        for (unsigned short x = -sizeDiv2; x < sizeDiv2 + 1; x++) {
//            tmp = source->data[]; //((float4 * ) & tmp[0], source->surface, (center.x + x) * sizeof(float4), (center.y + y), cudaBoundaryModeClamp);
//            result += tmp * weights.weights[(y + sizeDiv2) + (x + sizeDiv2) * weights.size];
//        }
//    }
//    target->data[];
////    surf2Dwrite<float4>(result, target->surface, center.x * sizeof(float4), center.y);
//}

#endif //DRTMCS_CONVOLUTIONS_CUH
