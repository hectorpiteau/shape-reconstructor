#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H
#include <glm/glm.hpp>
#include <surface_types.h>
#include <cuda_surface_types.h>
#include "CudaLinearVolume3D.cuh"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

using namespace glm;

/** ************************************************************************** */
//
// Helpers for indexing arrays, buffer and images all unified in order to
// be sure to have coherent indexing in all the code.
//
/** ************************************************************************** */

#define STBI_IMG_INDEX(X, Y, RESX) ((Y) * RESX * 4 + X * 4)
#define LINEAR_IMG_INDEX(X, Y, RESY) (X * RESY + Y)

#define VOLUME_INDEX(X, Y, Z, RES) (X * RES.y*RES.z + Y * RES.z + Z)

#define FLOAT4_NORM_TO_UCHAR4(F) make_uchar4( \
(unsigned char)__float2uint_rn(F.x * 255.0f), \
(unsigned char)__float2uint_rn(F.y * 255.0f), \
(unsigned char)__float2uint_rn(F.z * 255.0f), \
(unsigned char)__float2uint_rn(F.w * 255.0f))

#define FLOAT4_TO_UCHAR4(F) make_uchar4( \
(unsigned char)__float2uint_rn(F.x), \
(unsigned char)__float2uint_rn(F.y), \
(unsigned char)__float2uint_rn(F.z), \
(unsigned char)__float2uint_rn(F.w))


#define UCHAR4_TO_VEC3(UCH) vec3( \
__uint2float_rn(UCH.x), \
__uint2float_rn(UCH.y), \
__uint2float_rn(UCH.z))

#define UCHAR4_TO_VEC4(UCH) vec4( \
uint2float(UCH.x, cudaRoundNearest), \
uint2float(UCH.y, cudaRoundNearest), \
uint2float(UCH.z, cudaRoundNearest), \
uint2float(UCH.w, cudaRoundNearest))

#define VEC3_255_TO_UCHAR4(VEC) make_uchar4( \
(unsigned char)float2uint(VEC.x, cudaRoundNearest), \
(unsigned char)float2uint(VEC.y, cudaRoundNearest), \
(unsigned char)float2uint(VEC.z, cudaRoundNearest), 255)


#define VEC4_TO_UCHAR4(VEC) make_uchar4( \
__float2uint_rn(VEC.x), \
__float2uint_rn(VEC.y), \
__float2uint_rn(VEC.z), \
__float2uint_rn(VEC.w))

/** ************************************************************************** */
//
// Structs used to transfer data from CPU memory to GPU memory. Common between
// host and device.
//
/** ************************************************************************** */
struct VolumeDescriptor {
    /** min and max coordinates in world pos. */
    vec3 bboxMin;
    vec3 bboxMax;
    /** result of bbox min and max. gives the real world size. */
    vec3 worldSize;
    /** volume resolution. */
    ivec3 res;
    /** volume's data pointer. Indexed: [ x * (res.y*res.z) + y * (res.z) + z] */
    cell* data;
};

struct AdamOptimizerDescriptor {
    float epsilon;
    /** Step size. */
    float eta;
    /** Initialize default beta values. */
    vec2 beta;
    /** Gradient grid resolution. */
    ivec3 res;
    /** Gradients */
    VolumeDescriptor* grads;
    /** Adam gradients. */
    cell* adamG1;
    cell* adamG2;
    /** 3D Data to optimize. */
    cell* target;

    int iteration;
};

struct PlaneCutDescriptor {
    ushort axis; /** 0:X, y:1, 2:Z */
    float pos; /** Position along the axis. */
    vec3 min; /** The plane minimum along each axis. */
    vec3 max; /** The plane maximum along each axis. */

    /** The output texture that represent a screen overlay. */
    cudaSurfaceObject_t outSurface;
};

struct CursorPixel {
    ivec2 loc;
    vec4 value;
};



struct ImageDescriptor {
    /** image information */
    ivec2 imgRes;
    /** Image storage representation. */
    cudaSurfaceObject_t surface;
};

struct RayCasterDescriptor {
    /** Active zone min coordinates in pixels. */
    unsigned short minPixelX;
    unsigned short minPixelY;
    /** Active zone max coordinates in pixels. */
    unsigned short maxPixelX;
    unsigned short maxPixelY;
    // 6*ushort(16) = 96bits
    cudaSurfaceObject_t surface; // 64bits
    //total: 96 + 64 = 160 (alignment?)

    bool renderAllPixels;
};

struct CameraDescriptor {
    /** camera information */
    vec3 camPos;
    mat4 camInt;
    mat4 camExt;

    unsigned short width;
    unsigned short height;
};

struct GaussianWeightsDescriptor {
    float *weights = nullptr; /** Stored in column major. Smallest increment goes in the y direction. */
    /** Kernel's size. Must be odd. */

    /** Containing weights only from center to maximum size in one direction
     * because the distribution is fully symmetrical.
     * Array of size ceil(size/2.0). Ex: For kernel of size 3x3 => 2. */
    float* sw = nullptr;

    unsigned short size = 1;
    /** Kernel's dimension, either 2D or 3D. */
    unsigned short dim = 2;
    /** Kernel's stride in one dimension.
     * The size of the kernel from the center in one direction. For example
     * for a 3x3 kernel, ks=1. */
    unsigned short ks = 1;
};

struct LinearImageDescriptor {
    ivec2 res;
    unsigned char* data;
    vec3* vdata;
};

struct IntegrationRangeDescriptor {
    /**
     * column-major. Smallest increment goes in y+ direction.
     * Indexed : data[x * height + y];
     * */
    float2 *data;
    /** The data dimension. (width, height) */
    ivec2 dim;

    bool renderInTexture;
    cudaSurfaceObject_t surface;
};

struct BatchItemDescriptor{
    /** Combines Camera and Image Descriptors. */
    CameraDescriptor* cam;

    /** Ground-Truth image. */
    LinearImageDescriptor* img;
    ivec2 res;

    IntegrationRangeDescriptor* range;

    /** Data struct for storing the loss for each pixels. */
    vec4* loss;
    vec4* cpred;

    /** True for rendering the forward in the debugSurface. */
    bool debugRender;
    cudaSurfaceObject_t debugSurface;
};



struct BBoxDescriptor {
    vec3 min;
    vec3 max;
};

#endif //CUDA_COMMON_H