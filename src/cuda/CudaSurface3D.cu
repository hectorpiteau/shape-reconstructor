#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <iostream>

#include "CudaSurface3D.cuh"
#include "../utils/helper_cuda.h"

#define SURFACES_AMOUNT 5

// surface<void, cudaSurfaceType3D> surfaces3d[SURFACES_AMOUNT];

extern "C" void cuda_surface_wrapper()
{
    
}