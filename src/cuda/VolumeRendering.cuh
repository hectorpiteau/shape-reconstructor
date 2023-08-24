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
#include "Common.cuh"
#include "GPUData.cuh"


/**
 * @brief Volume Rendering Wrapper using Linear Memory Allocation :
 
 */
extern "C" void volume_rendering_wrapper(GPUData<RayCasterDescriptor>& raycaster, GPUData<CameraDescriptor>& camera, GPUData<DenseVolumeDescriptor>& volume);
extern "C" void sparse_volume_rendering_wrapper(GPUData<RayCasterDescriptor>& raycaster, GPUData<CameraDescriptor>& camera, GPUData<SparseVolumeDescriptor>* volume, GPUData<OneRayDebugInfoDescriptor> *debugRay);

extern "C" void batched_forward_wrapper(GPUData<BatchItemDescriptor>& item, GPUData<DenseVolumeDescriptor>& volume, GPUData<SuperResolutionDescriptor>& superRes);
extern "C" void batched_backward_wrapper(GPUData<BatchItemDescriptor>& item, GPUData<DenseVolumeDescriptor>& volume, GPUData<AdamOptimizerDescriptor>& adam, GPUData<SuperResolutionDescriptor>& superRes);
extern "C" void batched_forward_sparse_wrapper(GPUData<BatchItemDescriptor> &item, GPUData<SparseVolumeDescriptor> *volume, GPUData<SuperResolutionDescriptor>* superRes);
//extern "C" void batched_backward_sparse_wrapper(GPUData<BatchItemDescriptor>& item, GPUData<SparseVolumeDescriptor>* volume, GPUData<SparseAdamOptimizerDescriptor>& adam);
extern "C" void batched_backward_sparse_wrapper(GPUData<BatchItemDescriptor>* item, GPUData<SparseVolumeDescriptor>* volume, GPUData<SparseAdamOptimizerDescriptor>* adam, GPUData<SuperResolutionDescriptor>* superRes );

    extern "C" void volume_backward( GPUData<DenseVolumeDescriptor>* volume, GPUData<AdamOptimizerDescriptor>* adam);
extern "C" void sparse_volume_backward(GPUData<SparseVolumeDescriptor>* volume, GPUData<SparseAdamOptimizerDescriptor>* adam);
#endif // VOLUME_RENDERING_H
