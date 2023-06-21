#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "ShaderPipeline.hpp"
#include "Volume3D.hpp"
#include "Camera/Camera.hpp"
#include "../cuda/kernel.cuh"
#include "../cuda/RayCasterParams.cuh"

/** Cuda Kernels */
#include "../cuda/VolumeRendering.cuh"
#include "../cuda/PlaneCutRendering.cuh"

#include "../utils/helper_cuda.h"
#include "../cuda/GPUData.cuh"
#include "../cuda/Common.cuh"
#include "IntegrationRange.cuh"

class CudaTexture
{
private:
    bool m_isOpened = false;
    cudaGraphicsResource_t cuda_image_resource{};
    cudaArray_t cuda_image_array{};
    cudaSurfaceObject_t cuda_texture_surface{};
    cudaResourceDesc cuda_texture_resource_desc{};
    GLuint opengl_tex_cuda{};


    // Create 2D OpenGL texture in gl_tex and bind it to CUDA in cuda_tex
    void createGLTextureForCUDA(unsigned int size_x, unsigned int size_y)
    {
        // create an OpenGL texture
        glGenTextures(1, &opengl_tex_cuda);              // generate 1 texture
        glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda); // set it as current target
        // set basic texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        void *buf((void *)malloc(size_x * size_y * sizeof(GLubyte) * 4));

        // Specify 2D texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, (int)size_x, (int)size_y, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, buf);

        glBindTexture(GL_TEXTURE_2D, 0);
        free(buf);
        glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);

        // Register this texture with CUDA
        cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_image_resource, opengl_tex_cuda, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

        printf("cudaGraphicsGLRegisterImage error? [%d]:", err);

        if (err == cudaSuccess)
            printf("cudaSuccess\n");

        if (err == cudaErrorInvalidDevice)
            printf("cudaErrorInvalidDevice\n");

        if (err == cudaErrorInvalidValue)
            printf("cudaErrorInvalidValue\n");

        if (err == cudaErrorInvalidResourceHandle)
            printf("cudaErrorInvalidResourceHandle\n");

        if (err == cudaErrorUnknown)
            printf("cudaErrorUnknown\n");
    }

public:
    void Render(std::shared_ptr<ShaderPipeline> pipeline)
    {

    }

    [[nodiscard]] GLuint GetTex() const
    {
        return opengl_tex_cuda;
    }

    CudaTexture(uint width, uint height)
    {
        /** Init OpenGL Texture **/
        createGLTextureForCUDA(width, height);

        /** We don't want to use cudaMallocManaged here - since we definitely want Allocate CUDA memory for color output */
//        checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data));
    }

    ~CudaTexture()= default; //TODO:check if no memory leak here.

    void RunKernel(GPUData<RayCasterDescriptor>& raycasterDesc, GPUData<CameraDescriptor>& cameraDesc, GPUData<VolumeDescriptor>& volumeDesc)
    {
//        OpenSurface();
//        raycasterDesc.Host()->surface = cuda_texture_surface; // 64bits
//
//        /** kernel */
//        volume_rendering_wrapper(raycasterDesc, cameraDesc, volumeDesc, cuda_texture_surface);
//
//        CloseSurface();
    }


    void RunCUDAPlaneCut(GPUData<PlaneCutDescriptor>& planeCutDesc, GPUData<VolumeDescriptor>& volumeDesc, GPUData<CameraDescriptor>& cameraDesc ){
//        OpenSurface();
//
//        planeCutDesc.Host()->outSurface = cuda_texture_surface;
//
//        cameraDesc.ToDevice();
//        volumeDesc.ToDevice();
//        planeCutDesc.ToDevice();
//
//        /** kernel */
//        plane_cut_rendering_wrapper(planeCutDesc, volumeDesc, cameraDesc);
//        CloseSurface();
    }

    void RunCUDAIntegralRange(GPUData<IntegrationRangeDescriptor>& ranges, GPUData<CameraDescriptor>& camera,  BBoxDescriptor* bbox){
        OpenSurface();

        ranges.Host()->surface = cuda_texture_surface;
        ranges.ToDevice();

        /** kernel */
        integration_range_bbox_wrapper(camera, ranges.Device(), bbox);
        CloseSurface();
    }

    cudaSurfaceObject_t OpenSurface(){
        if(m_isOpened) return cuda_texture_surface;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_image_resource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_image_array, cuda_image_resource, 0, 0));
        // the resource Type says which things to set. See Documentation
        cuda_texture_resource_desc.resType = cudaResourceTypeArray;
        cuda_texture_resource_desc.res.array.array = cuda_image_array;
        // Create a surface Object
        checkCudaErrors(cudaCreateSurfaceObject(&cuda_texture_surface, &cuda_texture_resource_desc));

        m_isOpened = true;
        return cuda_texture_surface;
    }

    void CloseSurface(){
        if(!m_isOpened) return;
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_image_resource, 0));
        m_isOpened = false;
    }




};