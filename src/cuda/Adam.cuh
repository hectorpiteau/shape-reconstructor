//
// Created by hepiteau on 21/06/23.
//

#ifndef DRTMCS_ADAM_CUH
#define DRTMCS_ADAM_CUH

#include "Common.cuh"
#include "GPUData.cuh"

extern "C" void update_adam_wrapper(GPUData<AdamOptimizerDescriptor>* adam);
extern "C" void sparse_update_adam_wrapper(GPUData<SparseAdamOptimizerDescriptor>* adam);
extern "C" void zero_adam_wrapper(GPUData<AdamOptimizerDescriptor>* adam);
extern "C" void sparse_zero_adam_wrapper(GPUData<SparseAdamOptimizerDescriptor>* adam);
#endif //DRTMCS_ADAM_CUH
