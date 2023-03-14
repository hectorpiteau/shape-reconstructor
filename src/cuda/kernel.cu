#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.cuh"
#include <iostream>

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
	return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
	return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(b) << 16) | (int(g) << 8) | int(r);
}


__global__ void
cudaRender(unsigned int *g_odata, int imgw)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
	
    if(tx == 0){
        g_odata[y*imgw + x] = rgbToInt(255, 0, 0);
    }
    g_odata[y*imgw + x] = rgbToInt((tx*20)%255, 0, (ty*20)%255);
}



__global__ void kernel(uchar4 *tex_data, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        tex_data[index] = make_uchar4(threadIdx.x, 128, threadIdx.y, 255); // red
        // tex_data[index].x = 255; // red
        // tex_data[index].y = 0; // green
        // tex_data[index].z = 0; // blue
        // tex_data[index].w = 255; // alpha
    }
}

// extern "C"
void kernel_wrapper(uchar4 *d_ptr, int width, int height)
{
    int block_size = 16;
    dim3 grid_size((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);

    // for(int i=0; i<width; ++i){
    //     d_ptr[i] = make_uchar4(255, 0, 0, 255);
    // }
    kernel<<<grid_size, block_size>>>(d_ptr, width, height);

    cudaDeviceSynchronize();

    // std::cout << "kernel wrapped. " << std::endl;
}

extern "C"
void kernel_wrapper_2(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw)
{
	cudaRender<<<grid, block, sbytes>>>(g_odata, imgw);
}