/*
Author: Hector Piteau (hector.piteau@gmail.com)
CT3DVolume3D.cuh (c) 2023
Desc: A Cuda Volume3D that uses a Texture3D underneath.
Created:  2023-05-01T10:16:19.972Z
Modified: 2023-05-02T18:59:32.340Z
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <glm/glm.hpp>
#include <helper_cuda.h>

#include "../utils/helper_cuda.h"

// #define NUM_TEX 4

// const int SizeNoiseTest = 32;
// const int cubeSizeNoiseTest = SizeNoiseTest * SizeNoiseTest * SizeNoiseTest;
// static cudaTextureObject_t texNoise[NUM_TEX];

using namespace glm;

enum Location {
    DEVICE,
    HOST
};

class CT3DVolume3D
{
private:
    float *m_deviceDataArray;
    size_t m_sizeSide = 100;
    size_t m_sizeFull = m_sizeSide * m_sizeSide * m_sizeSide;
    cudaTextureObject_t m_texture;

    cudaArray *m_cudaArray;

    /** Is the array initialized, ready to be written. */
    bool m_isInitialized = false;
    /** Where the data is currently stored. */
    enum Location m_location = Location::HOST;

public:
    CT3DVolume3D(/* args */)
    {
        cudaMalloc((void **)&m_deviceDataArray, m_sizeFull * sizeof(float));

        // cudaArray Descriptor
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        checkCudaErrors(cudaMalloc3DArray(&m_cudaArray, &channelDesc, make_cudaExtent(m_sizeSide * sizeof(float), m_sizeSide, m_sizeSide), 0));
        
        // Array creation
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr(m_deviceDataArray, m_sizeSide * sizeof(float), m_sizeSide, m_sizeSide);
        copyParams.dstArray = m_cudaArray;
        copyParams.extent = make_cudaExtent(m_sizeSide, m_sizeSide, m_sizeSide);
        copyParams.kind = cudaMemcpyDeviceToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
        // Array creation End

        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = m_cudaArray;
        
        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));
        texDescr.normalizedCoords = false;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.addressMode[0] = cudaAddressModeClamp; // clamp
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;
        texDescr.readMode = cudaReadModeElementType;
        checkCudaErrors(cudaCreateTextureObject(&m_texture, &texRes, &texDescr, NULL));
    }

    ~CT3DVolume3D()
    {
        
    }

    __device__ void InitVolumeRGB(){

    }


    __device__ float Get(cudaTextureObject_t my_tex, const vec3& coords)
    {
        return tex3D<float>(my_tex,coords.x, coords.y, coords.z);
    }

    // void CreateTexture()
    // {

    //     float *d_NoiseTest; //Device Array with random floats

    //     cudaMalloc((void **)&d_NoiseTest, cubeSizeNoiseTest*sizeof(float)); // Allocation of device Array

    //     for (int i = 0; i < NUM_TEX; i++)
    //     {
    //         // curand Random Generator (needs compiler link -lcurand)
    //         curandGenerator_t gen;
    //         curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    //         curandSetPseudoRandomGeneratorSeed(gen, 1235ULL + i);
    //         curandGenerateUniform(gen, d_NoiseTest, cubeSizeNoiseTest); // writing data to d_NoiseTest
    //         curandDestroyGenerator(gen);

    //         // cudaArray Descriptor
    //         cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //         // cuda Array
    //         cudaArray *d_cuArr;
    //         checkCudaErrors(cudaMalloc3DArray(&d_cuArr, &channelDesc, make_cudaExtent(SizeNoiseTest * sizeof(float), SizeNoiseTest, SizeNoiseTest), 0));
    //         cudaMemcpy3DParms copyParams = {0};

    //         // Array creation
    //         copyParams.srcPtr = make_cudaPitchedPtr(d_NoiseTest, SizeNoiseTest * sizeof(float), SizeNoiseTest, SizeNoiseTest);
    //         copyParams.dstArray = d_cuArr;
    //         copyParams.extent = make_cudaExtent(SizeNoiseTest, SizeNoiseTest, SizeNoiseTest);
    //         copyParams.kind = cudaMemcpyDeviceToDevice;
    //         checkCudaErrors(cudaMemcpy3D(&copyParams));
    //         // Array creation End

    //         cudaResourceDesc texRes;
    //         memset(&texRes, 0, sizeof(cudaResourceDesc));
    //         texRes.resType = cudaResourceTypeArray;
    //         texRes.res.array.array = d_cuArr;
    //         cudaTextureDesc texDescr;
    //         memset(&texDescr, 0, sizeof(cudaTextureDesc));
    //         texDescr.normalizedCoords = false;
    //         texDescr.filterMode = cudaFilterModeLinear;
    //         texDescr.addressMode[0] = cudaAddressModeClamp; // clamp
    //         texDescr.addressMode[1] = cudaAddressModeClamp;
    //         texDescr.addressMode[2] = cudaAddressModeClamp;
    //         texDescr.readMode = cudaReadModeElementType;
    //         checkCudaErrors(cudaCreateTextureObject(&texNoise[i], &texRes, &texDescr, NULL));
    //     }
    // }
};