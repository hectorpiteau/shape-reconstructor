#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H
#include <glm/glm.hpp>

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

struct VolumeDescriptor {
    /** min and max coordinates in world pos. */
    vec3 bboxMin;
    vec3 bboxMax;

    /** result of bboxmin and max. gives the real world size. */
    vec3 worldSize;

    /** volume resolution. */
    ivec3 res;

    /** volume's data pointer. */
    float4* data;
};

struct ImageDescriptor {
    /** image informations */
    ivec2 imgRes;
};

struct RayCasterDescriptor {
    /* Camera's image plane dimensions.*/
    unsigned short width;
    unsigned short height;

    /** Active zone min coordinates in pixels. */
    unsigned short minPixelX;
    unsigned short minPixelY;

    /** Active zone max coordinates in pixels. */
    unsigned short maxPixelX;
    unsigned short maxPixelY;

    // 6*ushort(16) = 96bits

    cudaSurfaceObject_t surface; // 64bits    

    //total: 96 + 64 = 160 (alignment?)
};

struct CameraDescriptor {
    /** camera informations */
    vec3 camPos;
    mat4 camInt;
    mat4 camExt;

    unsigned short width;
    unsigned short height;
};

// __host__ void ToGPU(CameraDescriptor* dst, CameraDescriptor* src){
//     checkCudaErrors(
//             cudaMemcpy(dst, src, sizeof(struct CameraDescriptor), cudaMemcpyHostToDevice));
// }

// __host__ void ToGPU(ImageDescriptor* dst, ImageDescriptor* src){
//     checkCudaErrors(
//             cudaMemcpy(dst, src, sizeof(struct ImageDescriptor), cudaMemcpyHostToDevice));
// }

// __host__ void ToGPU(VolumeDescriptor* dst, VolumeDescriptor* src){
//     checkCudaErrors(
//             cudaMemcpy(dst, src, sizeof(struct VolumeDescriptor), cudaMemcpyHostToDevice));
// }




// CUDA_DEV int IsPointInVolume(const vec3& point){
//     if(any(lessThan(point, vec3(-0.5, -0.5, -0.5))) || any(greaterThan(point, vec3(0.5, 0.5, 0.5)))) return 0;
//     return 1;
// }

#endif //CUDA_COMMON_H