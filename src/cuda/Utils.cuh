#ifndef UTILS_CUH
#define UTILS_CUH

#include <glm/glm.hpp>
#include "Common.cuh"
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


CUDA_DEV inline glm::vec4 float4ToVec4(float4 a) {
    return {a.x, a.y, a.z, a.w};
}

CUDA_DEV inline glm::vec4 cellToVec4(cell x) {
#ifdef VOLUME_FP16
    auto a = __half22float2(x.rg);
    auto b = __half22float2(x.ba);
    return {a.x, a.y, b.x, b.y};
#elif defined VOLUME_FP32
    return {x.data.x, x.data.y, x.data.z, x.data.w};
#endif
}

CUDA_DEV inline float4 vec4ToFloat4(const glm::vec4 &a) {
    return make_float4(a.x, a.y, a.z, a.w);
}


/**
 * @brief Read inside a Dense Volume3D.
 *
 * @param data : A reference to a variable where the data will be written into.
 * @param pos : The sample position in world coordinate R3.
 * @param volume : The volume data storage.
 * @param resolution : The volume resolution in each direction.
 * @return bool :
 */
CUDA_DEV inline glm::vec4 ReadVolume(glm::vec3 &pos, VolumeDescriptor *volume) {
    /** Manual tri-linear interpolation. */
    auto local = (pos - volume->bboxMin) / volume->worldSize;
    glm::vec3 full_coords = local * glm::vec3(volume->res);
    full_coords -= vec3(0.5, 0.5, 0.5);
    glm::ivec3 min = glm::floor(full_coords); // first project [0,1] to [0, resolution], then take the floor index.
    glm::ivec3 max = glm::ceil(full_coords); // idem but to take the ceil index.
    min = glm::clamp(min, glm::ivec3(0, 0, 0), volume->res - 1);
    max = glm::clamp(max, glm::ivec3(0, 0, 0), volume->res - 1);

    glm::vec3 weights =  full_coords - vec3(min);

    glm::vec4 wx = vec4(weights.x);
    glm::vec4 wy = vec4(weights.y);
    glm::vec4 wz = vec4(weights.z);

    size_t x_step = volume->res.y * volume->res.z;
    size_t y_step = volume->res.z;

    /** Sample all around the pos point in the grid.  (8 voxels) */
    glm::vec4 c000 = cellToVec4(volume->data[min.x * x_step + min.y * y_step + min.z]); // back face
    glm::vec4 c001 = cellToVec4(volume->data[min.x * x_step + min.y * y_step + max.z]);
    glm::vec4 c010 = cellToVec4(volume->data[min.x * x_step + max.y * y_step + min.z]);
    glm::vec4 c011 = cellToVec4(volume->data[min.x * x_step + max.y * y_step + max.z]);

    glm::vec4 c100 = cellToVec4(volume->data[max.x * x_step + min.y * y_step + min.z]); // front face
    glm::vec4 c101 = cellToVec4(volume->data[max.x * x_step + min.y * y_step + max.z]);
    glm::vec4 c110 = cellToVec4(volume->data[max.x * x_step + max.y * y_step + min.z]);
    glm::vec4 c111 = cellToVec4(volume->data[max.x * x_step + max.y * y_step + max.z]);

    glm::vec4 c00 = glm::mix(c000, c100, wx);
    glm::vec4 c01 = glm::mix(c001, c101, wx);
    glm::vec4 c10 = glm::mix(c010, c110, wx);
    glm::vec4 c11 = glm::mix(c011, c111, wx);

    glm::vec4 c0 = glm::mix(c00, c10, wy);
    glm::vec4 c1 = glm::mix(c01, c11, wy);

    return glm::mix(c0, c1, wz);
}

CUDA_DEV inline void AtomicWriteVec4(glm::vec4 *addr, const glm::vec4 &data) {
    atomicAdd((float *) (addr), data.x);
    atomicAdd((float *) (addr + 1), data.y);
    atomicAdd((float *) (addr + 2), data.z);
    atomicAdd((float *) (addr + 3), data.w);
}

CUDA_DEV inline void AtomicWriteFloat4(float4 *addr, const glm::vec4 &data) {
    atomicAdd((float *) (&addr->x), data.x);
    atomicAdd((float *) (&addr->y), data.y);
    atomicAdd((float *) (&addr->z), data.z);
    atomicAdd((float *) (&addr->w), data.w);
}

CUDA_DEV inline void AtomicWriteCell(cell *addr, const glm::vec4 &data) {
    atomicAdd((float *) (&addr->data.x), data.x);
    atomicAdd((float *) (&addr->data.y), data.y);
    atomicAdd((float *) (&addr->data.z), data.z);
    atomicAdd((float *) (&addr->data.w), data.w);
}

/**
 * Write in the volume with tri-linear de-interpolation.
 *
 * @param pos : The position in world space coordinates.
 * @param volume : The volume to write into.
 * @return
 */
CUDA_DEV inline void WriteVolumeTRI(glm::vec3 &pos, VolumeDescriptor *volume, const glm::vec4 &value) {
    /** Manual tri-linear interpolation. */
    auto local = (pos - volume->bboxMin) / volume->worldSize;
    glm::vec3 full_coords = local * glm::vec3(volume->res);
    full_coords -= vec3(0.5, 0.5, 0.5);
    glm::ivec3 min = glm::floor(full_coords); // first project [0,1] to [0, resolution], then take the floor index.
    glm::ivec3 max = glm::ceil(full_coords); // idem but to take the ceil index.
    min = glm::clamp(min, glm::ivec3(0, 0, 0), volume->res - 1);
    max = glm::clamp(max, glm::ivec3(0, 0, 0), volume->res - 1);

    glm::vec3 w = full_coords - vec3(min);

    /** One Minus Weight */
    glm::vec3 omw = glm::vec3(1.0, 1.0, 1.0) - w;

    size_t x_step = volume->res.y * volume->res.z;
    size_t y_step = volume->res.z;

    /** Sample all around the pos point in the grid.  (8 voxels) */
    auto c000 = omw.x * omw.y * omw.z * value;
    auto c001 = omw.x * w.y * omw.z * value;
    auto c010 = omw.x * omw.y * w.z * value;
    auto c011 = omw.x * w.y * w.z * value;

    auto c100 = w.x * omw.y * omw.z * value;
    auto c101 = w.x * w.y * omw.z * value;
    auto c110 = w.x * omw.y * w.z * value;
    auto c111 = w.x * w.y * w.z * value;

    AtomicWriteCell(&volume->data[min.x * x_step + min.y * y_step + min.z], c000); // back face
    AtomicWriteCell(&volume->data[min.x * x_step + min.y * y_step + max.z], c001);
    AtomicWriteCell(&volume->data[min.x * x_step + max.y * y_step + min.z], c010);
    AtomicWriteCell(&volume->data[min.x * x_step + max.y * y_step + max.z], c011);

    AtomicWriteCell(&volume->data[max.x * x_step + min.y * y_step + min.z], c100); // front face
    AtomicWriteCell(&volume->data[max.x * x_step + min.y * y_step + max.z], c101);
    AtomicWriteCell(&volume->data[max.x * x_step + max.y * y_step + min.z], c110);
    AtomicWriteCell(&volume->data[max.x * x_step + max.y * y_step + max.z], c111);
}


#endif //UTILS_CUH
