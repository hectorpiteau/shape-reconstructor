#ifndef KERNEL_H
#define KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <glm/glm.hpp>

using namespace glm;

struct Ray
{
    vec3 origin;
    vec3 dir;
    float tmin, tmax;

};

struct VolumeData
{
    float4 data;
};



class RayCaster {
public:
    struct Ray GetRay(){
        struct Ray ray = {};
        return ray;
    }
};

// Forward declaration of CUDA render
extern "C" 
void volume_rendering_wrapper(RayCaster& rayCaster, cudaTextureObject_t &volume, float4* outTexture, size_t width, size_t height);

// void volume_rendering_wrapper(cudaTextureObject_t& volume, cudaSurfaceObject_t& outTexture, size_t width, size_t height, size_t depth);
#endif //KERNEL_H