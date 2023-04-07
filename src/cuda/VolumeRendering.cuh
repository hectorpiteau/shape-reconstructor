#ifndef KERNEL_H
#define KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>

// Forward declaration of CUDA render
extern "C" 
void volume_rendering_wrapper(cudaSurfaceObject_t& surf3d, size_t width, size_t height, size_t depth);


#endif //KERNEL_H