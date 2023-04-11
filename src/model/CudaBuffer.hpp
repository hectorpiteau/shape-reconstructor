#pragma once
#include "Buffer/Buffer.hpp"

template<typename T>
class CudaBuffer : public Buffer{
public:
    /**
     * @brief Construct a new Cuda Buffer object
     * 
     * @param size : The amount of object of type {T} in 
     * the buffer. 
     */
    CudaBuffer(std::size_t size){
        cudaMalloc((void**)&m_cudaBuffer, size*sizeof(T));
    }

    /**
     * @brief Transfer the data to the device (GPU).
     */
    void ToDevice(){

    }

    /**
     * @brief Transfer the data to host machine (CPU), alias 
     * RAM memory.
     */
    void ToHost(){

    }
    
    /** Remove copy constructor. */
    CudaBuffer(const CudaBuffer&) = delete;
    
    ~CudaBuffer() {
        cudaFree(m_cudaBuffer);
    }
    
    T operator[](std::size_t index){
        return m_cudaBuffer[index];
    }

    const T* GetPtr(){
        return m_cudaBuffer;
    }


    const T& Get(size_t index) {
        return m_cudaBuffer[index];
    }

    void Set(size_t index, const T& value) {
        m_cudaBuffer[index] = value;
    }

private:
    T* m_cudaBuffer;
    int m_length; 
};