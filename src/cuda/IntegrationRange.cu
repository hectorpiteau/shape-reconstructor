//
// Created by hpiteau on 14/06/23.
//

#include <iostream>
#include "IntegrationRange.cuh"
#include "SingleRayCaster.cuh"
#include "GPUData.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <cuda_gl_interop.h>



__global__ void
IntegrationRange(CameraDescriptor *camera, IntegrationRangeDescriptor *output_ranges, BBoxDescriptor *bbox) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= camera->width)
        return;
    if (y >= camera->height)
        return;

//    Ray ray = SingleRayCaster::GetRay(vec2(camera->width - x, camera->height - y), camera);
    Ray ray = SingleRayCaster::GetRay(vec2(camera->width - x, y), camera);

    float t_near = 0.0f, t_far = 0.0f;
    uchar4 element = make_uchar4(0, 0, 0, 255);
    /** Bbox - ray intersection. Testing min of the intersection of the 6 faces. */

    bool res = BBoxTminTmax(ray.origin, ray.dir, bbox->min, bbox->max, &t_near, &t_far);

    /** Write result in the output_ranges data array. */
    if (output_ranges->renderInTexture) {
        element.x = res ? (unsigned char)__float2uint_rn(t_far * 50.0f) : 0;
        element.y = res ? (unsigned char)__float2uint_rn(t_near * 50.0f) : 0;
        surf2Dwrite<uchar4>(element, output_ranges->surface, x * sizeof(uchar4), y);
    } else {
        output_ranges->data[x * output_ranges->dim.y + y] = make_float2(t_near, t_far);
    }
}

extern "C" void integration_range_bbox_wrapper(GPUData<CameraDescriptor>& camera, IntegrationRangeDescriptor *output_ranges,
                                               BBoxDescriptor *bbox) {
    dim3 threads(16, 16);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 blocks((camera.Host()->width + threads.x - 1) / threads.x,
                (camera.Host()->height + threads.y - 1) / threads.y);

    IntegrationRange<<<blocks, threads>>>(camera.Device(), output_ranges, bbox);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}
