#ifndef CUDA_SURFACE_3D_H
#define CUDA_SURFACE_3D_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include "../utils/helper_cuda.h"


/** 
 * This class can be used to hold a 3D grid of voxels data.
 * Each voxel is a float.
*/
class CudaSurface3D {
public:
    /**
     * @brief Construct a new Cuda Buffer object
     * 
     * @param width 
     * @param height 
     * @param depth 
     */
    CudaSurface3D(std::size_t width, std::size_t height, std::size_t depth) 
    : m_width(width), m_height(height), m_depth(depth){
        
        /** Create and allocate a cuda default array. It will store the surface's data. */
        m_channelDesc = cudaCreateChannelDesc<float>();
        const cudaExtent extent = make_cudaExtent(width, height, depth);   
        checkCudaErrors(
            cudaMalloc3DArray(&m_cudaArray, &m_channelDesc, extent, cudaArraySurfaceLoadStore)
        );

        /** Create the surface3D */
        memset(&m_resDesc, 0, sizeof(m_resDesc));
        m_resDesc.resType = cudaResourceTypeArray;
        m_resDesc.res.array.array = m_cudaArray; /** Assign the array of the surface to the previously declared array. */
        cudaCreateSurfaceObject(&m_surfaceObject, &m_resDesc);
        
    }

    void Bind();

    /**
     * @brief Transfer the data to the device (GPU).
     */
    void ToDevice(){

    }

    /**
     * @brief Transfer the data to host machine (CPU), alias 
     * RAM memory.
     */
    void ToHost(){};
    
    /** Remove copy constructor. */
    CudaSurface3D(const CudaSurface3D&) = delete;
    
    ~CudaSurface3D() {
        cudaDestroySurfaceObject(m_surfaceObject);
        cudaFreeArray(m_cudaArray);
    }

    cudaSurfaceObject_t& GetSurface(){
        return m_surfaceObject;
    }
    

private:
    
    int m_width; 
    int m_height; 
    int m_depth;

    cudaChannelFormatDesc m_channelDesc;

    struct cudaResourceDesc m_resDesc;
    cudaArray_t m_cudaArray; 
    
    cudaSurfaceObject_t m_surfaceObject;

};

#endif //CUDA_SURFACE_3D_H