#ifndef KERNEL_H
#define KERNEL_H



extern "C" void
// Forward declaration of CUDA render
volume_rendering_wrapper(surface<void, cudaSurfaceType3D>, Camera, int sbytes, unsigned int *g_odata, int imgw);


#endif //KERNEL_H