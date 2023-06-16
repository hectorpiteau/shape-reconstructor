#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H
#include <glm/glm.hpp>
#include <surface_types.h>
#include <cuda_surface_types.h>

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



struct AdamOptimizerDescriptor {
    float epsilon;
    /** Step size. */
    float eta;
    /** Initialize default beta values. */
    vec2 beta;
    /** Gradient grid resolution. */
    ivec3 res;
    /** Adam gradients. */
    float4* adamG1;
    float4* adamG2;
    /** 3D Data to optimize. */
    float4* target;
};

struct PlaneCutDescriptor {
    ushort axis; /** 0:X, y:1, 2:Z */
    float pos; /** Position along the axis. */
    vec3 min; /** The plane minimum along each axis. */
    vec3 max; /** The plane maximum along each axis. */

    /** The output texture that represent a screen overlay. */
    cudaSurfaceObject_t outSurface;
};

struct VolumeDescriptor {
    /** min and max coordinates in world pos. */
    vec3 bboxMin;
    vec3 bboxMax;
    /** result of bbox min and max. gives the real world size. */
    vec3 worldSize;
    /** volume resolution. */
    ivec3 res;
    /** volume's data pointer. Indexed: [ x * (res.y*res.z) + y * (res.z) + z] */
    float4* data;
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
    cudaArray* cdata;
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

    LinearImageDescriptor* img;

    IntegrationRangeDescriptor* range;

    /** True for rendering the forward in the debugSurface. */
    bool debugRender;
    cudaSurfaceObject_t debugSurface;
};



struct BBoxDescriptor {
    vec3 min;
    vec3 max;
};

#endif //CUDA_COMMON_H