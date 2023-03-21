#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <stdlib.h>


// #include "src/cuda/assert_cuda.h"
#include "interop.h"

struct pxl_interop
{
    // split GPUs?
    bool multi_gpu;

    // number of fbo's
    int count;
    int index;

    // w x h
    int width;
    int height;

    // GL buffers
    GLuint *fb;
    GLuint *rb;

    // CUDA resources
    cudaGraphicsResource_t *cgr;
    cudaArray_t *ca;
};

//
//
//

struct pxl_interop *
pxl_interop_create(const bool multi_gpu, const int fbo_count)
{
    struct pxl_interop *const interop = calloc(1, sizeof(*interop));

    interop->multi_gpu = multi_gpu;
    interop->count = fbo_count;
    interop->index = 0;

    // allocate arrays
    interop->fb = calloc(fbo_count, sizeof(*(interop->fb)));
    interop->rb = calloc(fbo_count, sizeof(*(interop->rb)));
    interop->cgr = calloc(fbo_count, sizeof(*(interop->cgr)));
    interop->ca = calloc(fbo_count, sizeof(*(interop->ca)));

    // render buffer object w/a color buffer
    glCreateRenderbuffers(fbo_count, interop->rb);

    // frame buffer object
    glCreateFramebuffers(fbo_count, interop->fb);

    // attach rbo to fbo
    for (int index = 0; index < fbo_count; index++)
    {
        glNamedFramebufferRenderbuffer(interop->fb[index],
                                       GL_COLOR_ATTACHMENT0,
                                       GL_RENDERBUFFER,
                                       interop->rb[index]);
    }

    // return it
    return interop;
}

void pxl_interop_destroy(struct pxl_interop *const interop)
{
    cudaError_t cuda_err;

    // unregister CUDA resources
    for (int index = 0; index < interop->count; index++)
    {
        if (interop->cgr[index] != NULL)
            cuda_err = cuda(GraphicsUnregisterResource(interop->cgr[index]));
    }

    // delete rbo's
    glDeleteRenderbuffers(interop->count, interop->rb);

    // delete fbo's
    glDeleteFramebuffers(interop->count, interop->fb);

    // free buffers and resources
    free(interop->fb);
    free(interop->rb);
    free(interop->cgr);
    free(interop->ca);

    // free interop
    free(interop);
}

cudaError_t pxl_interop_size_set(struct pxl_interop *const interop, const int width, const int height)
{
    cudaError_t cuda_err = cudaSuccess;

    // save new size
    interop->width = width;
    interop->height = height;

    // resize color buffer
    for (int index = 0; index < interop->count; index++)
    {
        // unregister resource
        if (interop->cgr[index] != NULL)
            cuda_err = cuda(GraphicsUnregisterResource(interop->cgr[index]));

        // resize rbo
        glNamedRenderbufferStorage(interop->rb[index], GL_RGBA8, width, height);

        // probe fbo status
        // glCheckNamedFramebufferStatus(interop->fb[index],0);

        // register rbo
        cuda_err = cuda(GraphicsGLRegisterImage(&interop->cgr[index],
                                                interop->rb[index],
                                                GL_RENDERBUFFER,
                                                cudaGraphicsRegisterFlagsSurfaceLoadStore |
                                                    cudaGraphicsRegisterFlagsWriteDiscard));
    }

    // map graphics resources
    cuda_err = cuda(GraphicsMapResources(interop->count, interop->cgr, 0));

    // get CUDA Array refernces
    for (int index = 0; index < interop->count; index++)
    {
        cuda_err = cuda(GraphicsSubResourceGetMappedArray(&interop->ca[index],
                                                          interop->cgr[index],
                                                          0, 0));
    }

    // unmap graphics resources
    cuda_err = cuda(GraphicsUnmapResources(interop->count, interop->cgr, 0));

    return cuda_err;
}

void pxl_interop_size_get(struct pxl_interop *const interop, int *const width, int *const height)
{
    *width = interop->width;
    *height = interop->height;
}

cudaError_t pxl_interop_map(struct pxl_interop *const interop, cudaStream_t stream)
{
    if (!interop->multi_gpu)
        return cudaSuccess;

    // map graphics resources
    return cuda(GraphicsMapResources(1, &interop->cgr[interop->index], stream));
}

cudaError_t pxl_interop_unmap(struct pxl_interop *const interop, cudaStream_t stream)
{
    if (!interop->multi_gpu)
        return cudaSuccess;

    return cuda(GraphicsUnmapResources(1, &interop->cgr[interop->index], stream));
}

cudaError_t pxl_interop_array_map(struct pxl_interop *const interop)
{
    //
    // FIXME -- IS THIS EVEN NEEDED?
    //

    cudaError_t cuda_err;

    // get a CUDA Array
    cuda_err = cuda(GraphicsSubResourceGetMappedArray(&interop->ca[interop->index],
                                                      interop->cgr[interop->index],
                                                      0, 0));
    return cuda_err;
}

cudaArray_const_t pxl_interop_array_get(struct pxl_interop *const interop)
{
    return interop->ca[interop->index];
}

int pxl_interop_index_get(struct pxl_interop *const interop)
{
    return interop->index;
}

void pxl_interop_swap(struct pxl_interop *const interop)
{
    interop->index = (interop->index + 1) % interop->count;
}

/*
static const GLenum attachments[] = { GL_COLOR_ATTACHMENT0 };
glInvalidateNamedFramebufferData(interop->fb[interop->index],1,attachments);
*/
void pxl_interop_clear(struct pxl_interop *const interop)
{

    const GLfloat clear_color[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glClearNamedFramebufferfv(interop->fb[interop->index], GL_COLOR, 0, clear_color);
}

void pxl_interop_blit(struct pxl_interop *const interop)
{
    glBlitNamedFramebuffer(interop->fb[interop->index], 0,
                           0, 0, interop->width, interop->height,
                           0, interop->height, interop->width, 0,
                           GL_COLOR_BUFFER_BIT,
                           GL_NEAREST);
}