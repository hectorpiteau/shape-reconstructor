#pragma once 

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "ShaderPipeline.hpp"
#include "../cuda/kernel.cuh"
#include "../utils/helper_cuda.h"

class CudaTexture
{
private:
    int m_width, m_height;

    void *cuda_dev_render_buffer;
    struct cudaGraphicsResource *cuda_tex_resource;
    GLuint opengl_tex_cuda;
    GLuint VBO, VAO, EBO;

    // Create 2D OpenGL texture in gl_tex and bind it to CUDA in cuda_tex
    void createGLTextureForCUDA(GLuint *gl_tex, struct cudaGraphicsResource **cuda_tex, unsigned int size_x, unsigned int size_y)
    {
        // create an OpenGL texture
        glGenTextures(1, gl_tex);              // generate 1 texture
        glBindTexture(GL_TEXTURE_2D, *gl_tex); // set it as current target
        // set basic texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE); // clamp t coordinate
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        void *buf((void *)malloc(size_x * size_y * sizeof(GLubyte) * 4));

        // Specify 2D texture
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA8UI_EXT, GL_UNSIGNED_BYTE, buf); //TODO: Maybe use 8bits int 
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, buf);

        glBindTexture(GL_TEXTURE_2D, 0);
        free(buf);
        glBindTexture(GL_TEXTURE_2D, *gl_tex);

        // Register this texture with CUDA
        cudaError_t err = cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        // SDK_CHECK_ERROR_GL();
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
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);

        pipeline->UseShader(); // we gonna use this compiled GLSL program
        glUniform1i(glGetUniformLocation(pipeline->m_programShader, "texture0"), 0);

        glBindVertexArray(VAO); // binding VAO automatically binds EBO
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0); // unbind VAO
    }

    GLuint GetTex(){
        return opengl_tex_cuda;
    }

    uint4 * GetCudaPtr(){
        return (uint4*)cuda_dev_render_buffer;
    }



    void RunCUDA()
    {
        /** calculate grid size */
        dim3 block(16, 16, 1);
        dim3 grid(m_width / block.x, m_height / block.y, 1);                               // 2D grid, every thread will compute a pixel
        kernel_wrapper_2(grid, block, 0, (unsigned int *)cuda_dev_render_buffer, m_width); // launch with 0 additional shared memory allocated

        /** We want to copy cuda_dev_render_buffer data to the texture */
        /** Map buffer objects to get CUDA device pointers */
        cudaArray *texture_ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

        int num_texels = m_width * m_height;
        int num_values = num_texels * 4;
        int size_tex_data = sizeof(GLubyte) * num_values;
        checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer, size_tex_data, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));
    }

    CudaTexture(int width, int height) : m_width(width), m_height(height)
    {
        /** Init OpenGL Texture **/
        createGLTextureForCUDA(&opengl_tex_cuda, &cuda_tex_resource, width, height);

        /** Init Cuda Buffers **/
        size_t size_tex_data;
        unsigned int num_texels;
        unsigned int num_values;

        num_texels = width * height;
        num_values = num_texels * 4;
        size_tex_data = sizeof(GLubyte) * num_values;

        /** We don't want to use cudaMallocManaged here - since we definitely want Allocate CUDA memory for color output */
        checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data));
    }

    ~CudaTexture()
    {
        // glDeleteRenderbuffers(m_depthBuffer);
        // if (pbo)
        // {
        //     cudaGraphicsUnregisterResource(cuda_pbo_resource);
        //     glDeleteBuffers(1, &pbo);
        //     glDeleteTextures(1, &tex);
        // }
    }

    void Compute()
    {
        // kernel_wrapper(m_d_ptr, m_width, m_height);
    }
};