#ifndef UTILS_CUH
#define UTILS_CUH

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


CUDA_DEV inline glm::vec4 float4ToVec4(float4 a) {
    return {a.x, a.y, a.z, a.w};
}

CUDA_DEV inline float4 vec4ToFloat4(const glm::vec4 &a) {
    return make_float4(a.x, a.y, a.z, a.w);
}


/**
 * @brief Read inside a Dense Volume3D.
 *
 * @param data : A reference to a variable where the data will be written into.
 * @param pos : The sample position in range [0, 1.0]^3.
 * @param volume : The volume data storage.
 * @param resolution : The volume resolution in each direction.
 * @return bool :
 */
CUDA_DEV inline glm::vec4 ReadVolume(glm::vec3 &pos, VolumeDescriptor *volume) {
    /** Manual tri-linear interpolation. */
    glm::vec3 full_coords = (pos + glm::vec3(0.5, 0.5, 0.5)) * glm::vec3(volume->res);
    glm::ivec3 min = glm::floor(full_coords); // first project [0,1] to [0, resolution], then take the floor index.
    glm::ivec3 max = glm::ceil(full_coords); // idem but to take the ceil index.
    min = glm::clamp(min, glm::ivec3(0, 0, 0), volume->res);
    max = glm::clamp(min, glm::ivec3(0, 0, 0), volume->res);

    glm::vec3 weights = glm::vec3(full_coords.x - (float) min.x, full_coords.y - (float) min.y,
                                  full_coords.z - (float) min.z);

    glm::vec4 wx = glm::vec4(weights.x, weights.x, weights.x, weights.x);
    glm::vec4 wy = glm::vec4(weights.y, weights.y, weights.y, weights.y);
    glm::vec4 wz = glm::vec4(weights.z, weights.z, weights.z, weights.z);

    size_t x_step = volume->res.y * volume->res.z;
    size_t y_step = volume->res.z;

    /** Sample all around the pos point in the grid.  (8 voxels) */
    glm::vec4 c000 = float4ToVec4(volume->data[min.x * x_step + min.y * y_step + min.z]); // back face
    glm::vec4 c001 = float4ToVec4(volume->data[min.x * x_step + max.y * y_step + min.z]);
    glm::vec4 c010 = float4ToVec4(volume->data[min.x * x_step + min.y * y_step + max.z]);
    glm::vec4 c011 = float4ToVec4(volume->data[min.x * x_step + max.y * y_step + max.z]);

    glm::vec4 c100 = float4ToVec4(volume->data[max.x * x_step + min.y * y_step + min.z]); // front face
    glm::vec4 c101 = float4ToVec4(volume->data[max.x * x_step + max.y * y_step + min.z]);
    glm::vec4 c110 = float4ToVec4(volume->data[max.x * x_step + min.y * y_step + max.z]);
    glm::vec4 c111 = float4ToVec4(volume->data[max.x * x_step + max.y * y_step + max.z]);

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
    atomicAdd((float *) (addr), data.x);
    atomicAdd((float *) (addr + 1), data.y);
    atomicAdd((float *) (addr + 2), data.z);
    atomicAdd((float *) (addr + 3), data.w);
}

/**
 * Write in the volume with tri-linear de-interpolation.
 * @param pos
 * @param volume
 * @return
 */
CUDA_DEV inline void WriteVolumeTRI(glm::vec3 &pos, VolumeDescriptor *volume, const glm::vec4 &value) {
    /** Manual tri-linear interpolation. */
    glm::vec3 full_coords = (pos + glm::vec3(0.5, 0.5, 0.5)) * glm::vec3(volume->res);
    glm::ivec3 min = glm::floor(full_coords); // first project [0,1] to [0, resolution], then take the floor index.
    glm::ivec3 max = glm::ceil(full_coords); // idem but to take the ceil index.
    min = glm::clamp(min, glm::ivec3(0, 0, 0), volume->res);
    max = glm::clamp(min, glm::ivec3(0, 0, 0), volume->res);

    glm::vec3 w = glm::vec3(full_coords.x - (float) min.x, full_coords.y - (float) min.y,
                            full_coords.z - (float) min.z);

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

    AtomicWriteFloat4(&volume->data[min.x * x_step + min.y * y_step + min.z], c000); // back face
    AtomicWriteFloat4(&volume->data[min.x * x_step + max.y * y_step + min.z], c001);
    AtomicWriteFloat4(&volume->data[min.x * x_step + min.y * y_step + max.z], c010);
    AtomicWriteFloat4(&volume->data[min.x * x_step + max.y * y_step + max.z], c011);

    AtomicWriteFloat4(&volume->data[max.x * x_step + min.y * y_step + min.z], c100); // front face
    AtomicWriteFloat4(&volume->data[max.x * x_step + max.y * y_step + min.z], c101);
    AtomicWriteFloat4(&volume->data[max.x * x_step + min.y * y_step + max.z], c110);
    AtomicWriteFloat4(&volume->data[max.x * x_step + max.y * y_step + max.z], c111);
}


#endif //UTILS_CUH
