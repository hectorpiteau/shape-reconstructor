/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRendering.cu (c) 2023
Desc: Volume rendering algorithms.
Created:  2023-04-13T12:33:22.433Z
Modified: 2023-04-17T11:37:50.055Z
*/

#include <glm/glm.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include "../utils/helper_cuda.h"
#include <device_launch_parameters.h>
#include <cmath>

#include "VolumeRendering.cuh"
#include "../model/RayCaster/Ray.h"
#include "SingleRayCaster.cuh"

static const float MIN_TRANSMITTANCE = 0.001f;

using namespace glm;

__device__ bool ReadVolume(struct VolumeData& data, vec3& pos, cudaTextureObject_t& volume){
    

    tex3D<float4>(&data.data, volume, pos.x, pos.y, pos.z);
    
    return true;
}


__device__ float tsdfToAlpha(float tsdf, float previousTsdf, float density){
    if(previousTsdf > tsdf){
        return (
            1.0f + exp(-density * previousTsdf)
        ) / (
            1.0f + exp(-density * tsdf)
        );
    }else{
        return 1.0f;
    }
}

// __device__ vec4 backward(struct Ray ray, cudaTextureObject_t& volume, vec3 dLoss_dLo, vec3 Lo){
//     float Tpartial = 1.0f;
//     vec3 Cpartial = vec3(0.0f, 0.0f, 0.0f);
//     float zeroCross = INFINITY; //0x7f800000; //std::numeric_limits<float>().infinity();

//     bool gradWritten = false;
// }


__device__ vec4 forward(Ray ray, cudaTextureObject_t& volume)
{
    /** Partial transmittance. */
    float Tpartial = 1.0f;
    /** Partial color. */
    vec3 Cpartial = vec3(0.0f, 0.0f, 0.0f);

    float previousTsdf = 1.0f;
    float step = 0.001f;
    float density = 1.0f;
    
    /** The ray's min must be strictly smaller than max. */
    if (ray.tmin < ray.tmax)
    {
        /** Travel through the ray from it's min to max. */
        for (float t = ray.tmin; t < ray.tmax; t += step)
        {
            vec3 worldPos = ray.origin + t * ray.dir;

            struct VolumeData data = {};

            // Read from input surface
            if(ReadVolume(data, worldPos, volume)){
                vec3 color = vec3(data.data.x, data.data.y, data.data.z);

                // sample exactly on the zero_crossing.
                // if(){}

                float alpha = tsdfToAlpha(data.data.w, previousTsdf, density);
                previousTsdf = data.data.w;

                Cpartial += color * (1.0f - alpha) * Tpartial;
                Tpartial *= alpha;

                if(Tpartial < MIN_TRANSMITTANCE){
                    Tpartial = 0.0f;
                    break;
                }
            }
        }
    }
    return vec4(Cpartial, Tpartial);
}

/** 2D kernel that project rays in the volume. */
__global__ void volumeRendering(RayCasterParams& params, cudaTextureObject_t& volume, float4* outTexture, size_t width, size_t height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width)return;
    if(y >= height)return;

    /** Compute Ray. */
    struct Ray ray = {
        .origin = vec3(0.0, 0.0, 0.0), 
        .dir = vec3(1.0, 0.0, 0.0), 
        .tmin = 0.0f, 
        .tmax = 1.0f
    };
    
    ray = SingleRayCaster::GetRay(vec2(x, y), params);

    /** Call forward. */
    vec4 result = forward(ray, volume);

    /** Store value in Out Memory. */
    outTexture[x * height + y].x = result.r;
    outTexture[x * height + y].y = result.g;
    outTexture[x * height + y].z = result.b;
}


/**
 * @brief 
 * 
 * It provide a ray for each pixel coordinates.
 * @param volume : A cuda Texture3D that contains the values of each texels.
 * @param outTexture : A 2D texture in which the volume rendering is rendered.
 * 
 */
extern "C" void volume_rendering_wrapper(RayCasterParams& params, cudaTextureObject_t &volume, float4* outTexture, size_t width, size_t height)
{
    /** Max 1024 per block. As each pixel is independant, may be useful to search for optimal size. */
    dim3 threadsperBlock(16, 16); 
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 numBlocks(
        (width + threadsperBlock.x - 1) / threadsperBlock.x, 
        (height + threadsperBlock.y - 1) / threadsperBlock.y
    );

    /** Call the main volumeRendering kernel. **/
    volumeRendering<<<numBlocks, threadsperBlock>>>(params, volume, outTexture, width, height);
    cudaDeviceSynchronize();
}