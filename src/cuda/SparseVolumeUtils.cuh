//
// Created by hepiteau on 24/07/23.
//

#ifndef DRTMCS_SPARSEVOLUME_UTILS_CUH
#define DRTMCS_SPARSEVOLUME_UTILS_CUH

#include <GPUData.cuh>
#include <Common.cuh>

extern "C" void sparse_volume_initialize(GPUData<SparseVolumeDescriptor>& volume);

#endif //DRTMCS_SPARSEVOLUME_UTILS_CUH
