#ifndef KERNEL_H
#define KERNEL_H

void kernel_wrapper(uchar4* d_ptr, int width, int height);

extern "C" void
// Forward declaration of CUDA render
kernel_wrapper_2(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw);


#endif //KERNEL_H