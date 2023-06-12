//
// Created by hpiteau on 08/06/23.
//

#ifndef DRTMCS_LOSSES_CUH
#define DRTMCS_LOSSES_CUH

#include <glm/glm.hpp>
#include "Common.cuh"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

CUDA_DEV inline glm::vec4 NeRFLoss(const Ray& ray, AdamOptimizerDescriptor* adam){

}

#endif //DRTMCS_LOSSES_CUH
