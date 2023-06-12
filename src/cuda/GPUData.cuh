#ifndef GPU_DATA_H
#define GPU_DATA_H
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include "../utils/helper_cuda.h"


template <class T>
class GPUData {
protected:
    T* m_host;
    T* m_device;

    bool m_isDeviceUpToDate;

public:
    GPUData() : m_host(nullptr), m_device(nullptr), m_isDeviceUpToDate(false) {
        m_host = (T*) malloc(sizeof(T));
        if(m_host == nullptr){
            std::cerr << "GPUData: Malloc error." << std::endl;
            exit(1);
        }
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
        m_isDeviceUpToDate = true;
    }

    void ToHost(){
        checkCudaErrors(
            cudaMemcpy((void*)m_host, (void*)m_device, sizeof(T), cudaMemcpyDeviceToHost)
        );
    }

};

#endif //GPU_DATA_H