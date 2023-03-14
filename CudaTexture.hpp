#ifndef CUDA_TEXTURE_H
#define CUDA_TEXTURE_H
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <iostream>

// #include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"

#include <memory>
#include "ShaderPipeline.hpp"
#include "src/cuda/kernel.cuh"
#include "helper_cuda.h"

class CudaTexture
{
private:
    int m_width, m_height;
    // GLuint m_bufferID;
    // GLuint m_texture;
    // GLuint m_depthBuffer;
    // GLuint m_framebuffer;
    // uchar4 *m_d_ptr;

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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, buf);

        glBindTexture(GL_TEXTURE_2D, 0);
        free(buf);
        glBindTexture(GL_TEXTURE_2D, *gl_tex);

        // Register this texture with CUDA
        cudaError_t err = cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        // SDK_CHECK_ERROR_GL();
        printf("cudaGraphicsGLRegisterImage error [%d]:", err);

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

        // glGenBuffers(1, &pbo);
        // glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        // glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * width * height * sizeof(GLubyte), 0, GL_STREAM_DRAW);
        // glGenTextures(1, &tex);
        // glBindTexture(GL_TEXTURE_2D, tex);
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        // cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);

        /** **************************************************************************************** */

        // /** Generate a texture id. */
        // glGenTextures(1, &m_texture);
        // glBindTexture(GL_TEXTURE_2D, m_texture);

        // // set basic parameters
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // // Create texture data (4-component unsigned byte)
        // // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        // struct cudaGraphicsResource *cuda_tex_resource;
        // cudaGraphicsGLRegisterImage(&cuda_tex_resource, m_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

        // cudaArray_t cuda_tex_array;
        // cudaGraphicsMapResources(1, &cuda_tex_resource);
        // cudaGraphicsSubResourceGetMappedArray(&cuda_tex_array, cuda_tex_resource, 0, 0);

        // size_t num_bytes;
        // cudaGraphicsResourceGetMappedPointer((void **)&m_d_ptr, &num_bytes, cuda_tex_resource);

        /** **************************************************************************************** */

        // Unbind the texture
        // glBindTexture(GL_TEXTURE_2D, 0);

        // glGenBuffers(1, &m_bufferID);
        // glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_bufferID);
        // glBufferData(GL_PIXEL_UNPACK_BUFFER, size, NULL, GL_STREAM_DRAW);

        // glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // glGenRenderbuffers( 1, &m_depthBuffer );
        // glBindRenderbuffer( GL_RENDERBUFFER, m_depthBuffer );

        // glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height );

        // // Unbind the depth buffer
        // glBindRenderbuffer( GL_RENDERBUFFER, 0 );

        // glGenFramebuffers( 1, &m_framebuffer );
        // glBindFramebuffer( GL_FRAMEBUFFER, m_framebuffer );

        // glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, GL_COLOR_ATTACHMENT0, 0 ); //colorAttachment0
        // glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, GL_DEPTH_ATTACHMENT ); //depthAttachment

        // int maxAttachments = 0;
        // glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS, &maxAttachments );

        // if(glCheckFramebufferStatus(m_framebuffer) != GL_FRAMEBUFFER_COMPLETE){
        //     std::cerr << "CUDATexture: Error creating the FrameBuffer." << std::endl;
        // }

        //         struct cudaGraphicsResource* resource;
        //         cudaError_t cudaGraphicsGLRegisterImage(
        //         &resource,
        //         GLuint image,
        //         GLenum target,
        //         unsigned int flags
        // )
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

    // GLuint GetTex() { return m_texture; }
};

#endif // CUDA_TEXTURE_H