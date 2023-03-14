#ifndef INTEROP_H
#define INTEROP_H

#include <cuda_runtime.h>
#include <stdbool.h>


struct pxl_interop * pxl_interop_create(const bool multi_gpu, const int fbo_count);
void pxl_interop_destroy(struct pxl_interop *const interop);


cudaError_t pxl_interop_size_set(struct pxl_interop *const interop, const int width, const int height);
void pxl_interop_size_get(struct pxl_interop *const interop, int *const width, int *const height);


cudaError_t pxl_interop_map(struct pxl_interop *const interop, cudaStream_t stream);
cudaError_t pxl_interop_unmap(struct pxl_interop *const interop, cudaStream_t stream);
cudaError_t pxl_interop_array_map(struct pxl_interop *const interop);


cudaArray_const_t pxl_interop_array_get(struct pxl_interop *const interop);
cudaStream_t pxl_interop_stream_get(struct pxl_interop *const interop);
int pxl_interop_index_get(struct pxl_interop *const interop);


void pxl_interop_swap(struct pxl_interop *const interop);
void pxl_interop_clear(struct pxl_interop *const interop);
void pxl_interop_blit(struct pxl_interop *const interop);

#endif //INTEROP_H