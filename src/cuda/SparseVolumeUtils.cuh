//
// Created by hepiteau on 24/07/23.
//

#ifndef RT3DRS_SPARSEVOLUME_UTILS_CUH
#define RT3DRS_SPARSEVOLUME_UTILS_CUH

#include <GPUData.cuh>
#include <Common.cuh>

extern "C" void sparse_volume_initialize(GPUData<SparseVolumeDescriptor>& volume);

#endif //RT3DRS_SPARSEVOLUME_UTILS_CUH
