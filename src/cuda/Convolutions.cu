//
// Created by hpiteau on 09/06/23.
//

#include <glm/glm.hpp>
#include <surface_functions.h>
#include <cuda_fp16.h>
#include "Common.cuh"
#include <iostream>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#define GAUSSIAN_CHECKS 1


//__global__ void Gaussian2D(GaussianWeightsDescriptor *gauss, ImageDescriptor *source, ImageDescriptor *target) {
//    /** If not using a dim2 kernel, it's an error. */
//#ifdef GAUSSIAN_CHECKS
//    if (gauss->dim != 2) return;
//#endif
//
//    unsigned short ks = gauss->ks;
//    /** Because we are using only the threads in the center, boundaries excluded, the
//     * coordinates in the image are proportional to blockDim minus the border size. */
//    unsigned int x = blockIdx.x * (blockDim.x - 2 * ks) + threadIdx.x;
//    unsigned int y = blockIdx.y * (blockDim.y - 2 * ks) + threadIdx.y;
//
//    /** Vector to accumulate the final result before writing back in global memory. */
//    glm::vec4 res = {0.0, 0.0, 0.0, 0.0};
//
//    /** Get shared memory Zone 1 & 2. */
//    extern __shared__ cell s[];
//    cell *z1 = s;
//    cell *z2 = &s[blockDim.y * blockDim.x];
//
//    /** Copy memory to shared space. */
//    surf2Dread((float4 *) &z1[threadIdx.y * blockDim.x + threadIdx.x], source->surface, (x) * sizeof(float4), (y),
//               cudaBoundaryModeClamp);
//    __syncthreads();
//
//    /** X-pass */
//    /** only the threads that are useful for the y-pass. So everyone but ones on the x-border. */
//    if (threadIdx.x > ks && threadIdx.x < (blockDim.x - ks)) {
//        int i;
//        for (i = -ks; i < ks + 1; i += 1) {
//            auto val = z1[(threadIdx.y) * blockDim.x + threadIdx.x + i];
//            auto w = gauss->sw[abs(ks)];
//            /** Write x-pass result in shared memory Zone-2 */
//            auto z2_index = threadIdx.y * blockDim.x + (threadIdx.x - ks);
//#ifdef VOLUME_FP16
//            z2[z2_index].rg.x = fma(val.rg.x, w, z2[z2_index].rg.x);
//            z2[z2_index].rg.y = __hfma(val.rg.y, w, z2[z2_index].rg.y);
//            z2[z2_index].ba.x = __hfma(val.ba.x, w, z2[z2_index].ba.x);
//            z2[z2_index].ba.y = __hfma(val.ba.y, w, z2[z2_index].ba.y);
//#elif defined VOLUME_FP32
//            z2[z2_index].x += val.x * w;
//            z2[z2_index].y += val.y * w;
//            z2[z2_index].z += val.z * w;
//            z2[z2_index].w += val.w * w;
//#endif
//
//        }
//    }
//
//    /**  make sure that all threads have finished computing the x-pass*/
//    __syncthreads();
//
//    /** Only use useful threads, the middle ones without borders. */
//    if (threadIdx.x > ks && threadIdx.x < (blockDim.x - ks) && threadIdx.y > ks && threadIdx.y < (blockDim.y - ks)) {
//        int i;
//        for (i = -ks; i < ks + 1; i += 1) {
//            /** Read values from shared-memory Zone 2. */
//            auto val = z2[(threadIdx.y + i) * blockDim.x + threadIdx.x];
//            auto w = gauss->sw[abs(ks)];
//
//#ifdef VOLUME_FP16
//            res.x = __hfma(val.rg.x, w, res.x);
//            res.y = __hfma(val.rg.y, w, res.y);
//            res.z = __hfma(val.ba.x, w, res.z);
//            res.w = __hfma(val.ba.y, w, res.w);
//#elif defined VOLUME_FP32
//            res.x += val.x * w;
//            res.y += val.y * w;
//            res.z += val.z * w;
//            res.w += val.w * w;
//#endif
//        }
//        /** Write y-pass result in target image global memory. */
//        float4 test = make_float4(res.x, res.y, res.z, res.w);
//        surf2Dwrite(test, target->surface, (unsigned int)(x * sizeof(float4)), y);
//    }
//}
//
//__global__ void Gaussian3D(GaussianWeightsDescriptor *gauss, VolumeDescriptor *source, VolumeDescriptor *target) {
//#ifdef GAUSSIAN_CHECKS
//    if (gauss->dim != 3) return;
//#endif
//    unsigned short ks = floor((float) gauss->size / 2.0f);
//
//    unsigned int x = blockIdx.x * (blockDim.x - 2 * ks) + threadIdx.x;
//    unsigned int y = blockIdx.y * (blockDim.y - 2 * ks) + threadIdx.y;
//    unsigned int z = blockIdx.z * (blockDim.z - 2 * ks) + threadIdx.z;
//
//    glm::vec4 res{};
//
//    /** Get shared memory Zone 1 & 2. */
//#ifdef VOLUME_FP16
//    extern __shared__ cell s[];
//    cell *z1 = &s[0];
//    cell *z2 = &s[blockDim.y * blockDim.x * blockDim.z];
//
//#elif defined VOLUME_FP32
//    extern __shared__ float4 s[];
//    float4 *z1 = &s[0];
//    float4 *z2 = &s[blockDim.y * blockDim.x * blockDim.z];
//
//#endif
//
//    /** Copy memory to shared space. */
//    auto z1_index = threadIdx.x * (blockDim.y * blockDim.z) + threadIdx.y * blockDim.z + threadIdx.z;
//    auto source_index = x * (source->res.y * source->res.z) + y * (source->res.z) + z;
//
//#ifdef VOLUME_FP16
//    z1[z1_index] = source->data[source_index];
//#elif defined VOLUME_FP32
//    z1[z1_index] = source->data[source_index];
//#endif
//    __syncthreads();
//
//    /** X-pass */
//    /** only the threads that are useful for the y-pass. So everyone but ones on the x-border. */
//    if (threadIdx.x > ks && threadIdx.x < (blockDim.x - ks)) {
//        int i;
//        for (i = -ks; i < ks + 1; i += 1) {
//            auto val = z1[(threadIdx.x + i) * (blockDim.y * blockDim.z) + threadIdx.y * blockDim.z + threadIdx.z];
//
//#ifdef VOLUME_FP16
//            auto w = __float2half(gauss->sw[abs(ks)]);
//#elif defined VOLUME_FP32
//            auto w = gauss->sw[abs(ks)];
//#endif
//            /** Write x-pass result in shared memory Zone-2 */
//            auto z2_index = (threadIdx.x) * (blockDim.y * blockDim.z) + threadIdx.y * blockDim.z + threadIdx.z;
//
//#ifdef VOLUME_FP16
//            z2[z2_index].rg.x = __hfma(val.rg.x, w, z2[z2_index].rg.x);
//            z2[z2_index].rg.y = __hfma(val.rg.y, w, z2[z2_index].rg.y);
//            z2[z2_index].ba.x = __hfma(val.ba.x, w, z2[z2_index].ba.x);
//            z2[z2_index].ba.y = __hfma(val.ba.y,w, z2[z2_index].ba.y);
//#elif defined VOLUME_FP32
//            z2[z2_index].x += val.x * w;
//            z2[z2_index].y += val.y * w;
//            z2[z2_index].z += val.z * w;
//            z2[z2_index].w += val.w * w;
//#endif
//
//        }
//    }
//
//    /**  make sure that all threads have finished computing the x-pass*/
//    __syncthreads();
//
//    /** Y-Pass */
//    /** Only use useful threads, the middle ones without borders. */
//    if (threadIdx.x > ks && threadIdx.x < (blockDim.x - ks)
//        && threadIdx.y > ks && threadIdx.y < (blockDim.y - ks)) {
//        int i;
//        for (i = -ks; i < ks + 1; i += 1) {
//            /** Read values from shared-memory Zone 2. */
//            auto z2_index = (threadIdx.x) * (blockDim.y * blockDim.z) + (threadIdx.y + i) * blockDim.z + threadIdx.z;
//            auto val = z2[z2_index];
//            auto w = gauss->sw[abs(ks)];
//            auto z1_index_y = (threadIdx.x) * (blockDim.y * blockDim.z) + (threadIdx.y + i) * blockDim.z + threadIdx.z;
//#ifdef VOLUME_FP16
//            z1[z1_index_y].rg.x = __hfma(val.rg.x, w, z1[z1_index_y].rg.x);
//            z1[z1_index_y].rg.y = __hfma(val.rg.y, w, z1[z1_index_y].rg.y);
//            z1[z1_index_y].ba.x = __hfma(val.ba.x, w, z1[z1_index_y].ba.x);
//            z1[z1_index_y].ba.y = __hfma(val.ba.y,w, z1[z1_index_y].ba.y);
//#elif defined VOLUME_FP32
//            z1[z1_index_y].x += val.x * w;
//            z1[z1_index_y].y += val.y * w;
//            z1[z1_index_y].z += val.z * w;
//            z1[z1_index_y].w += val.w * w;
//#endif
//
//        }
//    }
//
//    /**  make sure that all threads have finished computing the y-pass*/
//    __syncthreads();
//
//    /** Z-Pass */
//    /** Only use useful threads, the middle ones without borders. */
//    if (threadIdx.x > ks && threadIdx.x < (blockDim.x - ks)
//        && threadIdx.y > ks && threadIdx.y < (blockDim.y - ks)
//        && threadIdx.z > ks && threadIdx.z < (blockDim.z - ks)) {
//        int i;
//        for (i = -ks; i < ks + 1; i += 1) {
//            /** Read values from shared-memory Zone 2. */
//            auto z2_index = (threadIdx.x) * (blockDim.y * blockDim.z) + threadIdx.y * blockDim.z + (threadIdx.z + i);
//            auto val = z2[z2_index];
//            auto w = gauss->sw[abs(ks)];
//
//#ifdef VOLUME_FP16
//            res.x = __hfma(val.rg.x, w, res.x);
//            res.y = __hfma(val.rg.y, w, res.y);
//            res.z = __hfma(val.ba.x, w, res.z);
//            res.w = __hfma(val.ba.y,w, res.w);
//#elif defined VOLUME_FP32
//            res.x += val.x * w;
//            res.y += val.y * w;
//            res.z += val.z * w;
//            res.w += val.w * w;
//#endif
//        }
//        /** Write y-pass result in target image global memory. */
//        target->data[x * (source->res.y * source->res.z) + y * (source->res.z) + z] = make_float4(res.x, res.y, res.z,
//                                                                                                  res.w);
//    }
//}
//
//void Gaussian2DWrapper(GaussianWeightsDescriptor *gauss, ImageDescriptor *source, ImageDescriptor *target) {
//    int ks = 1;
//
//    /** Max 1024 per block. As each pixel is independent, may be useful to search for optimal size. */
//    dim3 threadsPerBlock(16, 16);
//
//    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
//    dim3 numBlocks(
//            ceil(target->imgRes.x / (threadsPerBlock.x - 2 * ks)),
//            ceil(target->imgRes.y / (threadsPerBlock.y - 2 * ks))
//    );
//
//    /** Use one cell per thread as threads contains borders already.*/
//    size_t shared_memory_space =
//            threadsPerBlock.x * threadsPerBlock.y * sizeof(float4)              /** for the first temporary area.*/
//            + (threadsPerBlock.x - 2 * ks) * threadsPerBlock.y * sizeof(float4);  /** for the second temporary area*/
//
//    /** Call the main volumeRendering kernel. **/
//    Gaussian2D<<<numBlocks, threadsPerBlock, shared_memory_space>>>(gauss, source, target);
//
//    /** Get last error after rendering. */
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
//    }
//}
//
//void Gaussian3DWrapper(GaussianWeightsDescriptor *gauss, VolumeDescriptor *source, VolumeDescriptor *target) {
//    int ks = 1;
//
//    /** Max 1024 per block. As each pixel is independent, may be useful to search for optimal size. */
//    dim3 threadsPerBlock(8, 8, 8);
//
//    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
//    dim3 numBlocks(
//            ceil(target->res.x / (threadsPerBlock.x - 2 * ks)),
//            ceil(target->res.y / (threadsPerBlock.y - 2 * ks)),
//            ceil(target->res.z / (threadsPerBlock.z - 2 * ks))
//    );
//
//    /** Use one cell per thread as threads contains borders already.*/
//    size_t shared_memory_space =
//            threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
//            sizeof(float4)              /** for the first temporary area.*/
//            + threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
//              sizeof(float4);  /** for the second temporary area*/
//
//    /** Call the main volumeRendering kernel. **/
//    Gaussian3D<<<numBlocks, threadsPerBlock, shared_memory_space>>>(gauss, source, target);
//
//    /** Get last error after rendering. */
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
//    }
//}