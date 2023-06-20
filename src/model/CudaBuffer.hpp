#pragma once
#include "Buffer/Buffer.hpp"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif


template<typename T>
class CudaBuffer : public Buffer<T> {
private:
    T* m_cudaBuffer;
    size_t m_size;

    bool m_loaded;
public:
    /**
     * @brief Construct a new Cuda Buffer object
     * 
     * @param size : The amount of object of type {T} in 
     * the buffer. 
     */
    explicit CudaBuffer(std::size_t size) : m_size(size), m_loaded(true){
        checkCudaErrors(
            cudaMalloc((void**)&m_cudaBuffer, size*sizeof(T))
        );
    }

    explicit CudaBuffer() : m_cudaBuffer(nullptr), m_size(0), m_loaded(false){}

    CUDA_HOSTDEV void Allocate(size_t size){
        if(m_loaded) return;
        m_size = size;
        checkCudaErrors(
                cudaMalloc((void**)&m_cudaBuffer, size*sizeof(T))
        );
        m_loaded = true;
    }

    CUDA_HOSTDEV void Free(size_t size){
        m_size = 0;
        cudaFree(m_cudaBuffer);
        m_loaded = false;
    }

    CUDA_HOSTDEV [[nodiscard]] bool IsLoaded() const {
        return m_loaded;
    }

    /**
     * Get the pointer to gpu data location.
     * Available on both host and device but can only accessed on device.
     *
     * @return A pointer to the memory location on gpu.
     */
    CUDA_HOSTDEV T* Device(){
        return m_cudaBuffer;
    }

    /**
     * Get the buffer's size.
     * @return The amount of items in the buffer.
     */
    CUDA_HOSTDEV [[nodiscard]] size_t Size() const {
        return m_size;
    }

    /**
     * Get the size in memory in Bytes.
     *
     * @return The memory allocated in bytes.
     */
    CUDA_HOSTDEV [[nodiscard]] size_t MemorySize() const {
        return m_size * sizeof(T);
    }




//    /**
//     * @brief Transfer the data to the device (GPU).
//     */
//    CUDA_HOST void ToDevice(){
//        checkCudaErrors(
//                cudaMemcpy(m_gpuData, m_hostData, m_size, cudaMemcpyHostToDevice)
//        );
//    }
//
//    /**
//     * @brief Transfer the data to host machine (CPU), alias
//     * RAM memory.
//     */
//    void ToHost(){
//        checkCudaErrors(
//                cudaMemcpy(m_gpuData, m_hostData, m_size, cudaMemcpyDeviceToHost)
//        );
//    }
    
    /** Remove copy constructor. */
    CudaBuffer(const CudaBuffer&) = delete;
    
    ~CudaBuffer() {
        cudaFree(m_cudaBuffer);
    }
    
    CUDA_DEV T operator[](std::size_t index){
        return m_cudaBuffer[index];
    }

    CUDA_DEV const T& Get(size_t index) {
        return m_cudaBuffer[index];
    }

    CUDA_DEV void Set(size_t index, const T& value) {
        m_cudaBuffer[index] = value;
    }


};