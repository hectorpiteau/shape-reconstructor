/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRendering.cu (c) 2023
Desc: Volume rendering algorithms.
Created:  2023-04-13T12:33:22.433Z
Modified: 2023-05-11T22:28:51.324Z
*/

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include "../utils/helper_cuda.h"
// #include "../utils/helper_math.h"
#include <device_launch_parameters.h>
#include <cmath>

#include "VolumeRendering.cuh"
#include "../model/RayCaster/Ray.h"
#include "SingleRayCaster.cuh"

static const float MIN_TRANSMITTANCE = 0.001f;

using namespace glm;

__device__ bool ReadVolumeLinear(struct VolumeData& data, vec3& pos, float4* volume, ivec3 res){
    ivec3 fpos = (ivec3)floor(pos);
    
    data.data = volume[
          fpos.x * (res.y * res.z) 
        + fpos.y * (res.z) 
        + fpos.z
    ];
    
    return true;
}

// __device__ float4 TriLinearInterpolation(){
//     float4 c000 =  
//     float4 c001 = 
//     float4 c010 = 
//     float4 c011 =
//     float4 c100 =
//     float4 c101 =
//     float4 c110 =
//     float4 c111 =
// }

inline __device__ vec4 float4ToVec4(float4 a)
{
    return vec4(a.x, a.y, a.z, a.w);
}

inline __device__ float4 vec4ToFloat4(vec4 a)
{
    return make_float4(a.x, a.y, a.z, a.w);
}

/**
 * @brief 
 * 
 * @param data : A reference to a variable where the data will be written into. 
 * @param pos : The sample position in range [0, 1.0]^3.
 * @param volume : The volume data storage.
 * @param resolution : The volume resolution in each direction.
 * @return bool : 
 */
__device__ bool ReadVolume(struct VolumeData& data, vec3& pos, float4* volume, const ivec3& resolution){
    /** Manual tri-linear interpolation. */
    vec3 full_coords = pos * vec3(resolution);
    ivec3 min = floor(full_coords); // first project [0,1] to [0, resolution], then take the floor index.
    ivec3 max = ceil(full_coords);  // idem but to take the ceil index. 
    vec3 weights = vec3(full_coords.x - min.x, full_coords.y - min.y, full_coords.z - min.z);

    vec4 wx = vec4(weights.x,weights.x,weights.x,weights.x);
    vec4 wy = vec4(weights.y,weights.y,weights.y,weights.y);
    vec4 wz = vec4(weights.z,weights.z,weights.z,weights.z);

    size_t x_step = resolution.y*resolution.z;
    size_t y_step = resolution.z;

    /** Sample all around the pos point in the grid.  (8 voxels) */
    vec4 c000 = float4ToVec4(volume[min.x * x_step + min.y * y_step  + min.z]); //back face
    vec4 c001 = float4ToVec4(volume[min.x * x_step + max.y * y_step  + min.z]);  
    vec4 c010 = float4ToVec4(volume[min.x * x_step + min.y * y_step  + max.z]);
    vec4 c011 = float4ToVec4(volume[min.x * x_step + max.y * y_step  + max.z]);
    
    vec4 c100 = float4ToVec4(volume[max.x * x_step + min.y * y_step  + min.z]); //front face
    vec4 c101 = float4ToVec4(volume[max.x * x_step + max.y * y_step  + min.z]);
    vec4 c110 = float4ToVec4(volume[max.x * x_step + min.y * y_step  + max.z]);
    vec4 c111 = float4ToVec4(volume[max.x * x_step + max.y * y_step  + max.z]);

    vec4 c00 = mix(c000, c100, wx);
    vec4 c01 = mix(c001, c101, wx);
    vec4 c10 = mix(c010, c110, wx);
    vec4 c11 = mix(c011, c111, wx);

    vec4 c0 = mix(c00, c10, wy);
    vec4 c1 = mix(c01, c11, wy);

    data.data = vec4ToFloat4(mix(c0, c1, wz));
    
    return true;
}

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

__device__ vec4 forward(Ray ray, float4* volume, const ivec3& resolution)
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
            if(ReadVolume(data, worldPos, volume, resolution)){
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

__global__ void volumeRenderingUI8(RayCasterParams& params, float4* volume, const ivec3& volumeResolution, uint4* outTexture, size_t width, size_t height)
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
    vec4 result = forward(ray, volume, volumeResolution);

    /** Scale up. */
    result *= 255.0f;

    /** Store value in Out Memory. */
    outTexture[x * height + y] = make_uint4(result.x, result.y, result.z, result.w);
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


__global__ void testFillBlue(RayCasterParams& params, float4* volume, float4* outTexture, size_t width, size_t height)
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
    vec4 result = vec4(0.0, 0.0, 1.0, 1.0);

    /** Store value in Out Memory. */
    outTexture[x * height + y].x = result.r;
    outTexture[x * height + y].y = result.g;
    outTexture[x * height + y].z = result.b;
}


extern "C" void volume_rendering_wrapper_linea_ui8(RayCasterParams& params, float4* volume, const ivec3& volumeResolution, uint4* outTexture, size_t width, size_t height){
    /** Max 1024 per block. As each pixel is independant, may be useful to search for optimal size. */
    dim3 threadsperBlock(16, 16);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 numBlocks(
        (width + threadsperBlock.x - 1) / threadsperBlock.x, 
        (height + threadsperBlock.y - 1) / threadsperBlock.y
    );

    /** Call the main volumeRendering kernel. **/
    volumeRenderingUI8<<<numBlocks, threadsperBlock>>>(params, volume, volumeResolution, outTexture, width, height);
    cudaDeviceSynchronize();
}

extern "C" void volume_rendering_wrapper_linear(RayCasterParams& params, float4* volume, float4 *outTexture, size_t width, size_t height){
    std::cout << "not implemented yet." << std::endl;
}