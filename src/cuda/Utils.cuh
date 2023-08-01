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
 * Return the evaluation of a general 2D gaussian function
 * centered on 0 with g(0) = 1.
 *
 * @param x : The x coordinate on the plane.
 * @param y : The y coordinate on the plane.
 * @return exp(- x*x - y*y).
 */
CUDA_HOSTDEV inline float GeneralGaussian2D(float x, float y) {
    return exp(-x * x - y * y);
}

/**
 * Return the evaluation of a custom 2D gaussian function defined by sigma and mu parameters.
 *
 * @param x : The x coordinate on the plane.
 * @param y : The y coordinate on the plane.
 * @param sig : The standard deviation.
 * @param mu : The mean.
 * @return The evaluation of the gaussian function with custom parameters.
 */
CUDA_HOSTDEV inline float CustomGaussian2D(float x, float y, float sig, float mu) {
    return 1.0f / (sig * sqrt(2 * M_PI)) *
           exp((-(x - mu) * (x - mu)) / (2 * sig * sig) + (-(y - mu) * (y - mu)) / (2 * sig * sig));
}


/**
 * @brief Read inside a Dense DenseVolume3D.
 *
 * @param data : A reference to a variable where the data will be written into.
 * @param pos : The sample position in world coordinate R3.
 * @param volume : The volume data storage.
 * @param resolution : The volume resolution in each direction.
 * @return bool :
 */
CUDA_DEV inline glm::vec4 ReadVolume(glm::vec3 &pos, DenseVolumeDescriptor *volume, size_t *indices) {
    /** Manual tri-linear interpolation. */
    auto local = (pos - volume->bboxMin) / volume->worldSize;
    glm::vec3 full_coords = local * glm::vec3(volume->res);
//    full_coords -= vec3(0.5, 0.5, 0.5);
    glm::ivec3 min = glm::floor(full_coords); // first project [0,1] to [0, resolution], then take the floor index.
    glm::ivec3 max = min + ivec3(1); //glm::ceil(full_coords); // idem but to take the ceil index.
    min = glm::clamp(min, glm::ivec3(0, 0, 0), volume->res - 1);
    max = glm::clamp(max, glm::ivec3(0, 0, 0), volume->res - 1);

    glm::vec3 weights = full_coords - vec3(min);

    glm::vec4 wx = vec4(weights.x);
    glm::vec4 wy = vec4(weights.y);
    glm::vec4 wz = vec4(weights.z);

    size_t x_step = volume->res.y * volume->res.z;
    size_t y_step = volume->res.z;

    indices[0] = min.x * x_step + min.y * y_step + min.z;
    indices[1] = min.x * x_step + min.y * y_step + max.z;
    indices[2] = min.x * x_step + max.y * y_step + min.z;
    indices[3] = min.x * x_step + max.y * y_step + max.z;

    indices[4] = max.x * x_step + min.y * y_step + min.z;
    indices[5] = max.x * x_step + min.y * y_step + max.z;
    indices[6] = max.x * x_step + max.y * y_step + min.z;
    indices[7] = max.x * x_step + max.y * y_step + max.z;

    /** Sample all around the pos point in the grid.  (8 voxels) */
    glm::vec4 c000 = cellToVec4(volume->data[indices[0]]); // back face
    glm::vec4 c001 = cellToVec4(volume->data[indices[1]]);
    glm::vec4 c010 = cellToVec4(volume->data[indices[2]]);
    glm::vec4 c011 = cellToVec4(volume->data[indices[3]]);

    glm::vec4 c100 = cellToVec4(volume->data[indices[4]]); // front face
    glm::vec4 c101 = cellToVec4(volume->data[indices[5]]);
    glm::vec4 c110 = cellToVec4(volume->data[indices[6]]);
    glm::vec4 c111 = cellToVec4(volume->data[indices[7]]);

    glm::vec4 c00 = glm::mix(c000, c100, wx);
    glm::vec4 c01 = glm::mix(c001, c101, wx);
    glm::vec4 c10 = glm::mix(c010, c110, wx);
    glm::vec4 c11 = glm::mix(c011, c111, wx);

    glm::vec4 c0 = glm::mix(c00, c10, wy);
    glm::vec4 c1 = glm::mix(c01, c11, wy);

    return glm::mix(c0, c1, wz);
}

CUDA_DEV inline unsigned int SparseVolumeGetDataIndex(ivec3 coords, SparseVolumeDescriptor *volume) {
    /** Locate the coarser cell in the stage0. */
    auto s0_tmp = vec3(coords) / vec3(4);

    auto s0_coords = ivec3(floor(s0_tmp));

    unsigned int s0index = STAGE0_INDEX(s0_coords.x, s0_coords.y, s0_coords.z, volume->initialResolution );

    auto s0_cell = volume->stage0[s0index];

    /** If the cell is not active then nothing is inside. return. */
    if (s0_cell.active == false) return INF;

    /** If the voxel is not empty, locate the cell in the stage1. */
    if (s0_cell.index >= volume->stage1Size) return INF;
    auto s1_cell = volume->stage1[s0_cell.index];

    auto previous_res = volume->initialResolution;

    /** While its not a leaf, traverse the tree. */
    while (!s1_cell.is_leaf) {
        auto current_coords = vec3(coords) / vec3(4);
        auto s1_tmp = floor((round(current_coords) - floor(current_coords)) * 4.0f);
        auto tmp_ind = SHIFT_INDEX_4x4x4(ivec3(s1_tmp));
        if (tmp_ind >= 64) return INF;
        auto index = s1_cell.indexes[tmp_ind];
        if (index >= volume->stage1Size) return INF;
        s1_cell = volume->stage1[index];
        previous_res = previous_res * 4;
    }

    /** The cell is a leaf so the indexes correspond to the data buffer now. */
    /** Let find the last voxel in the last block of 4x4x4. */
    auto current_coords = vec3(coords) / vec3(4);
    auto s1_tmp = glm::floor((glm::round(current_coords) - glm::floor(current_coords)) * 4.0f);
    s1_tmp = glm::clamp(s1_tmp, vec3(0), vec3(4 - 1));

    auto tmp_ind = SHIFT_INDEX_4x4x4(ivec3(s1_tmp));
    if (tmp_ind >= 64) return INF;

    return s1_cell.indexes[tmp_ind];
}


CUDA_DEV inline void SparseVolumeSet(vec3 coords, vec4 value, SparseVolumeDescriptor *volume) {
    auto index = SparseVolumeGetDataIndex(coords, volume);
    if (index == INF) return;
    volume->data[index].data = make_float4(value.x, value.y, value.z, value.w);
}

CUDA_DEV inline void SparseVolumeAtomicSet(vec3 coords, vec4 value, SparseVolumeDescriptor *volume) {
    auto index = SparseVolumeGetDataIndex(coords, volume);
    if (index == INF) return;
    volume->data[index].data = make_float4(value.x, value.y, value.z, value.w);
}

CUDA_DEV inline cell SparseVolumeGet(ivec3 coords, SparseVolumeDescriptor *volume) {
    auto index = SparseVolumeGetDataIndex(coords, volume);
    /** Check if it points to infinity, then there is no data. */
    if (index == INF) return {.data = make_float4(0.0, 0.0, 0.0, 0.0)};
    return volume->data[index];
}

CUDA_DEV inline glm::vec4 ReadVolume(glm::vec3 &pos, SparseVolumeDescriptor *volume, size_t *indices) {
    /** Manual tri-linear interpolation. */
    auto local = (pos - volume->bboxMin) / volume->worldSize;
    glm::vec3 full_coords = local * glm::vec3(volume->res);
    glm::ivec3 min = glm::floor(full_coords);
    glm::ivec3 max = min + ivec3(1);

    min = glm::clamp(min, glm::ivec3(0), volume->res - ivec3(1));
    max = glm::clamp(max, glm::ivec3(0), volume->res - ivec3(1));

    glm::vec3 weights = full_coords - vec3(min);

    glm::vec4 wx = vec4(weights.x);
    glm::vec4 wy = vec4(weights.y);
    glm::vec4 wz = vec4(weights.z);

//    auto res = vec3(min.x % 32, min.y %32, min.z %32) / glm::vec3(32); //volume->res
//    return vec4(res, 1.0);

    indices[0] = SparseVolumeGetDataIndex(vec3(min.x, min.y, min.z), volume); //min.x * x_step + min.y * y_step + min.z;
    if (indices[0] == INF) return vec4(1.0, 0.0, 0.0, 1.0);
    indices[1] = SparseVolumeGetDataIndex(vec3(min.x, min.y, max.z), volume); //min.x * x_step + min.y * y_step + max.z;
    if (indices[1] == INF) return vec4(0.0, 0.0, 1.0, 1.0);
    indices[2] = SparseVolumeGetDataIndex(vec3(min.x, max.y, min.z), volume); //min.x * x_step + max.y * y_step + min.z;
    if (indices[2] == INF) return vec4(1.0, 0.0, 1.0, 1.0);
    indices[3] = SparseVolumeGetDataIndex(vec3(min.x, max.y, max.z), volume); //min.x * x_step + max.y * y_step + max.z;
    if (indices[3] == INF) return vec4(1.0, 1.0, 0.0, 1.0);

    indices[4] = SparseVolumeGetDataIndex(vec3(max.x, min.y, min.z), volume); //max.x * x_step + min.y * y_step + min.z;
    if (indices[4] == INF) return vec4(0.0, 0.5, 0.0, 1.0);
    indices[5] = SparseVolumeGetDataIndex(vec3(max.x, min.y, max.z), volume); //max.x * x_step + min.y * y_step + max.z;
    if (indices[5] == INF) return vec4(0.0, 0.6, 0.0, 1.0);
    indices[6] = SparseVolumeGetDataIndex(vec3(max.x, max.y, min.z), volume); //max.x * x_step + max.y * y_step + min.z;
    if (indices[6] == INF) return vec4(0.0, 0.7, 0.0, 1.0);
    indices[7] = SparseVolumeGetDataIndex(vec3(max.x, max.y, max.z), volume); //max.x * x_step + max.y * y_step + max.z;
    if (indices[7] == INF) return vec4(0.0, 0.8, 0.0, 1.0);

    /** Sample all around the pos point in the grid.  (8 voxels) */
    glm::vec4 c000 = cellToVec4(volume->data[indices[0]]); // back face
    glm::vec4 c001 = cellToVec4(volume->data[indices[1]]);
    glm::vec4 c010 = cellToVec4(volume->data[indices[2]]);
    glm::vec4 c011 = cellToVec4(volume->data[indices[3]]);

    glm::vec4 c100 = cellToVec4(volume->data[indices[4]]); // front face
    glm::vec4 c101 = cellToVec4(volume->data[indices[5]]);
    glm::vec4 c110 = cellToVec4(volume->data[indices[6]]);
    glm::vec4 c111 = cellToVec4(volume->data[indices[7]]);

    glm::vec4 c00 = glm::mix(c000, c100, wx);
    glm::vec4 c01 = glm::mix(c001, c101, wx);
    glm::vec4 c10 = glm::mix(c010, c110, wx);
    glm::vec4 c11 = glm::mix(c011, c111, wx);

    glm::vec4 c0 = glm::mix(c00, c10, wy);
    glm::vec4 c1 = glm::mix(c01, c11, wy);

    return glm::mix(c0, c1, wz);
}

CUDA_DEV inline glm::vec4 ReadVolumeNearest(glm::vec3 &pos, DenseVolumeDescriptor *volume) {
    /** Manual tri-linear interpolation. */
    auto local = (pos - volume->bboxMin) / volume->worldSize;
    glm::vec3 full_coords = local * glm::vec3(volume->res);
    full_coords -= vec3(0.5, 0.5, 0.5);
    glm::ivec3 nearest = glm::round(full_coords); // first project [0,1] to [0, resolution], then take the floor index.
    nearest = glm::clamp(nearest, glm::ivec3(0, 0, 0), volume->res - 1);
    size_t x_step = volume->res.y * volume->res.z;
    size_t y_step = volume->res.z;
    return cellToVec4(volume->data[nearest.x * x_step + nearest.y * y_step + nearest.z]);
}

CUDA_DEV inline glm::vec4 ReadVolumeNearest(glm::vec3 &pos, SparseVolumeDescriptor *volume) {
    /** Manual tri-linear interpolation. */
    auto local = (pos - volume->bboxMin) / volume->worldSize;
    glm::vec3 full_coords = local * glm::vec3(volume->res);
    full_coords -= vec3(0.5, 0.5, 0.5);
    glm::ivec3 nearest = glm::round(full_coords); // first project [0,1] to [0, resolution], then take the floor index.
    nearest = glm::clamp(nearest, glm::ivec3(0, 0, 0), volume->res - 1);

    return cellToVec4(volume->data[SparseVolumeGetDataIndex(nearest, volume)]);
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
CUDA_DEV inline void WriteVolumeTRI(glm::vec3 &pos, DenseVolumeDescriptor *volume, const glm::vec4 &value, size_t *indices,
                                    AdamOptimizerDescriptor *adam) {
    /** Manual tri-linear interpolation. */
    auto local = (pos - volume->bboxMin) / volume->worldSize;
    glm::vec3 full_coords = local * glm::vec3(volume->res);
    glm::ivec3 min = glm::floor(full_coords);
    min = glm::clamp(min, glm::ivec3(0, 0, 0), volume->res - 1);

    glm::vec3 w = full_coords - vec3(min);

    /** One Minus Weight */
    glm::vec3 omw = glm::vec3(1.0, 1.0, 1.0) - w;

    /** Sample all around the pos point in the grid.  (8 voxels) */

    for (int i = 0; i < adam->amountOfGradientsToWrite; ++i) {
        auto index = adam->writeGradientIndexes[i];

        switch (index) {
            case 0:
                // c000
                AtomicWriteCell(&volume->data[indices[0]], omw.x * omw.y * omw.z * value); // back face
                break;
            case 1:
                //c001
                AtomicWriteCell(&volume->data[indices[1]], omw.x * w.y * omw.z * value);
                break;
            case 2:
                //c010
                AtomicWriteCell(&volume->data[indices[2]], omw.x * omw.y * w.z * value);
                break;
            case 3:
                //c011
                AtomicWriteCell(&volume->data[indices[3]], omw.x * w.y * w.z * value);
                break;
            case 4:
                //c100
                AtomicWriteCell(&volume->data[indices[4]], w.x * omw.y * omw.z * value); // front face
                break;
            case 5:
                //c101
                AtomicWriteCell(&volume->data[indices[5]], w.x * w.y * omw.z * value);
                break;
            case 6:
                //c110
                AtomicWriteCell(&volume->data[indices[6]], w.x * omw.y * w.z * value);
                break;
            case 7:
                //c111
                AtomicWriteCell(&volume->data[indices[7]], w.x * w.y * w.z * value);
                break;
        }
    }
}

/**
 * Write in the volume with tri-linear de-interpolation.
 *
 * @param pos : The position in world space coordinates.
 * @param volume : The volume to write into.
 * @return
 */
CUDA_DEV inline void WriteVolumeTRI(glm::vec3 &pos, SparseVolumeDescriptor *volume, const glm::vec4 &value, size_t *indices,
                                    SparseAdamOptimizerDescriptor *adam) {
    /** Manual tri-linear interpolation. */
    auto local = (pos - volume->bboxMin) / volume->worldSize;
    glm::vec3 full_coords = local * glm::vec3(volume->res);
    glm::ivec3 min = glm::floor(full_coords);
    min = glm::clamp(min, glm::ivec3(0, 0, 0), volume->res - 1);

    glm::vec3 w = full_coords - vec3(min);

    /** One Minus Weight */
    glm::vec3 omw = glm::vec3(1.0, 1.0, 1.0) - w;

    /** Sample all around the pos point in the grid.  (8 voxels) */
    for (int i = 0; i < adam->amountOfGradientsToWrite; ++i) {
        auto index = adam->writeGradientIndexes[i];
        switch (index) {
            case 0:
                // c000
                AtomicWriteCell(&volume->data[indices[0]], omw.x * omw.y * omw.z * value); // back face
                break;
            case 1:
                //c001
                AtomicWriteCell(&volume->data[indices[1]], omw.x * w.y * omw.z * value);
                break;
            case 2:
                //c010
                AtomicWriteCell(&volume->data[indices[2]], omw.x * omw.y * w.z * value);
                break;
            case 3:
                //c011
                AtomicWriteCell(&volume->data[indices[3]], omw.x * w.y * w.z * value);
                break;
            case 4:
                //c100
                AtomicWriteCell(&volume->data[indices[4]], w.x * omw.y * omw.z * value); // front face
                break;
            case 5:
                //c101
                AtomicWriteCell(&volume->data[indices[5]], w.x * w.y * omw.z * value);
                break;
            case 6:
                //c110
                AtomicWriteCell(&volume->data[indices[6]], w.x * omw.y * w.z * value);
                break;
            case 7:
                //c111
                AtomicWriteCell(&volume->data[indices[7]], w.x * w.y * w.z * value);
                break;
        }
    }
}

CUDA_HOSTDEV inline bool test_bit_4x4x4(unsigned long bits, unsigned int shift) {
    unsigned long bit_to_test = 1 << shift;
    return bits && bit_to_test;
};


#endif //UTILS_CUH
