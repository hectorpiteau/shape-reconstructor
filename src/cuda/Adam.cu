//
// Created by hepiteau on 21/06/23.
//
#include <glm/glm.hpp>
#include "Adam.cuh"
#include "Common.cuh"
#include "Utils.cuh"
#include "GPUData.cuh"


__global__ void UpdateAdam(AdamOptimizerDescriptor* adam){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x > adam->res.x || y > adam->res.y || z > adam->res.z) return;

    auto target = float4ToVec4(adam->target[VOLUME_INDEX(x,y,z, adam->res)].data);
    auto grad = float4ToVec4(adam->grads->data[VOLUME_INDEX(x,y,z, adam->res)].data);
    auto g1 = float4ToVec4(adam->adamG1[VOLUME_INDEX(x,y,z, adam->res)].data);
    auto g2 = float4ToVec4(adam->adamG2[VOLUME_INDEX(x,y,z, adam->res)].data);

    auto m_g1 = adam->beta.x * g1 + (1.0f - adam->beta.x) * grad;
    auto v_g2 = adam->beta.y * g2 + (1.0f - adam->beta.y)*( grad * grad) ;

    auto m_dw_corr = m_g1 / (float)(1.0f - pow(adam->beta.x, adam->iteration)); // adam->iteration
    auto v_dw_corr = v_g2 / (float)(1.0f - pow(adam->beta.y,adam->iteration)); // adam->iteration

    /** Update target volume weights. */
    auto adamRes = (adam->eta * ( m_dw_corr /(sqrt( v_dw_corr ) + adam->epsilon)));
    auto tmp = target - clamp(adamRes, vec4(-0.01), vec4(0.01));
    tmp = clamp(tmp, vec4(0.0), vec4(1.0));
    tmp.w = clamp(tmp.w, 0.0f, 0.99f);

    adam->target[VOLUME_INDEX(x,y,z, adam->res)].data = vec4ToFloat4(tmp);

    /** Update adam gradients. */
    adam->adamG1[VOLUME_INDEX(x,y,z, adam->res)].data = vec4ToFloat4(m_g1);
    adam->adamG2[VOLUME_INDEX(x,y,z, adam->res)].data = vec4ToFloat4(v_g2);
}

__global__ void ZeroAdam(AdamOptimizerDescriptor* adam){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x > adam->res.x || y > adam->res.y || z > adam->res.z) return;

    adam->grads->data[VOLUME_INDEX(x,y,z,adam->res)].data = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

}

extern "C" void update_adam_wrapper(GPUData<AdamOptimizerDescriptor>* adam){
    dim3 threads(8, 8, 8);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 blocks((adam->Host()->res.x + threads.x - 1) / threads.x,
                (adam->Host()->res.y + threads.y - 1) / threads.y,
                (adam->Host()->res.z + threads.z - 1) / threads.z);

    UpdateAdam<<<blocks, threads>>>(adam->Device());

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(update_adam_wrapper) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}

extern "C" void zero_adam_wrapper(GPUData<AdamOptimizerDescriptor>* adam){
    dim3 threads(8, 8, 8);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 blocks((adam->Host()->res.x + threads.x - 1) / threads.x,
                (adam->Host()->res.y + threads.y - 1) / threads.y,
                (adam->Host()->res.z + threads.z - 1) / threads.z);

    ZeroAdam<<<blocks, threads>>>(adam->Device());

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "(zero_adam_wrapper) ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}