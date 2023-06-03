#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

template <class T>
class GPUData {
private:
    T* m_host;
    T* m_device;

    bool isDeviceUpToDate;

public:
    GPUData() : m_host(nullptr), m_device(nullptr), isDeviceUpToDate(false) {
        m_host = (T*) malloc(sizeof(T));
        
        checkCudaErrors(cudaMalloc((void **)&m_device, sizeof(T)));
    }

    GPUData(const GPUData&) = delete;
    
    ~GPUData(){
        if(m_host != nullptr) free(m_host);
        if(m_device != nullptr) cudaFree(m_device);
    }

    T* Device() {return m_device;}
    T* Host() {return m_host;}

    void ToDevice(){
        checkCudaErrors(
            cudaMemcpy((void*)m_device, (void*)m_host, sizeof(T), cudaMemcpyHostToDevice)
        );
    }

    void ToHost(){
        checkCudaErrors(
            cudaMemcpy((void*)m_host, (void*)m_device, sizeof(T), cudaMemcpyDeviceToHost)
        );
    }

};