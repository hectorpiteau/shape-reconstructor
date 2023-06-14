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

    Ray ray = SingleRayCaster::GetRay(vec2(camera->width - x, camera->height - y), camera);

    float t_near = -1.0f, t_far = -1.0f;
    uchar4 element = make_uchar4(0, 0, 0, 255);
    /** Bbox - ray intersection. Testing min of the intersection of the 6 faces. */
    /** 1 */
    if (VectorPlaneIntersectionExist(ray.dir, vec3(1, 0, 0))) {
        auto point = VectorPlaneIntersection(ray.origin, ray.dir, vec3(bbox->max.x, 0.0f, 0.0f), vec3(1, 0, 0));
        if (point.y < bbox->max.y && point.y > bbox->min.x
            && point.z < bbox->max.z && point.z > bbox->min.z) {
            auto tmp = VectorPlaneIntersectionT(ray.origin, ray.dir, vec3(bbox->max.x, 0.0f, 0.0f), vec3(1, 0, 0));
            if(t_far < 0.0f) t_far = tmp;
            else t_far = tmp > t_far ? tmp : t_far;
            if(t_near < 0.0f) t_near = tmp;
            else t_near = tmp < t_near ? tmp : t_near;
        }
    }
    /** 2 */
    if (VectorPlaneIntersectionExist(ray.dir, vec3(-1, 0, 0))) {
        auto point = VectorPlaneIntersection(ray.origin, ray.dir, vec3(bbox->min.x, 0.0f, 0.0f), vec3(-1, 0, 0));
        if (point.y < bbox->max.y && point.y > bbox->min.x
            && point.z < bbox->max.z && point.z > bbox->min.z) {
            auto tmp = VectorPlaneIntersectionT(ray.origin, ray.dir, vec3(bbox->min.x, 0.0f, 0.0f), vec3(-1, 0, 0));
            if(t_far < 0.0f) t_far = tmp;
            else t_far = tmp > t_far ? tmp : t_far;
            if(t_near < 0.0f) t_near = tmp;
            else t_near = tmp < t_near ? tmp : t_near;
        }
    }
    /** 3 */
    if (VectorPlaneIntersectionExist(ray.dir, vec3(0, 0, -1))) {
        auto point = VectorPlaneIntersection(ray.origin, ray.dir, vec3(0.0f, 0.0f, bbox->min.z), vec3(0, 0, -1));
        if (point.y < bbox->max.y && point.y > bbox->min.x
            && point.x < bbox->max.x && point.x > bbox->min.x) {
            auto tmp = VectorPlaneIntersectionT(ray.origin, ray.dir, vec3(0.0f, 0.0f, bbox->min.z), vec3(0, 0, -1));
            if(t_far < 0.0f) t_far = tmp;
            else t_far = tmp > t_far ? tmp : t_far;
            if(t_near < 0.0f) t_near = tmp;
            else t_near = tmp < t_near ? tmp : t_near;
        }
    }
    /** 4 */
    if (VectorPlaneIntersectionExist(ray.dir, vec3(0, 0, 1))) {
        auto point = VectorPlaneIntersection(ray.origin, ray.dir, vec3(0.0f, 0.0f, bbox->max.z), vec3(0, 0, 1));
        if (point.y < bbox->max.y && point.y > bbox->min.x
            && point.x < bbox->max.x && point.x > bbox->min.x) {
            auto tmp = VectorPlaneIntersectionT(ray.origin, ray.dir, vec3(0.0f, 0.0f, bbox->max.z), vec3(0, 0, 1));
            if(t_far < 0.0f) t_far = tmp;
            else t_far = tmp > t_far ? tmp : t_far;
            if(t_near < 0.0f) t_near = tmp;
            else t_near = tmp < t_near ? tmp : t_near;
        }
    }
    /** 5 */
    if (VectorPlaneIntersectionExist(ray.dir, vec3(0, -1, 0))) {
        auto point = VectorPlaneIntersection(ray.origin, ray.dir, vec3(0.0f, bbox->min.y, 0.0f), vec3(0, -1, 0));
        if (point.z < bbox->max.z && point.z > bbox->min.z
            && point.x < bbox->max.x && point.x > bbox->min.x) {
            auto tmp = VectorPlaneIntersectionT(ray.origin, ray.dir, vec3(0.0f, bbox->min.y, 0.0f), vec3(0, -1, 0));
            if(t_far < 0.0f) t_far = tmp;
            else t_far = tmp > t_far ? tmp : t_far;
            if(t_near < 0.0f) t_near = tmp;
            else t_near = tmp < t_near ? tmp : t_near;
        }
    }
    /** 6 */
    if (VectorPlaneIntersectionExist(ray.dir, vec3(0, 1, 0))) {
        auto point = VectorPlaneIntersection(ray.origin, ray.dir, vec3(0.0f, bbox->max.y, 0.0f), vec3(0, 1, 0));
        if (point.z < bbox->max.z && point.z > bbox->min.z
            && point.x < bbox->max.x && point.x > bbox->min.x) {
            auto tmp = VectorPlaneIntersectionT(ray.origin, ray.dir, vec3(0.0f, bbox->max.y, 0.0f), vec3(0, 1, 0));
            if(t_far < 0.0f) t_far = tmp;
            else t_far = tmp > t_far ? tmp : t_far;
            if(t_near < 0.0f) t_near = tmp;
            else t_near = tmp < t_near ? tmp : t_near;
        }
    }
//    vec3 planeNormal = vec3(1.0, 0.0, 0.0);
//    auto test = VectorPlaneIntersectionExist(ray.dir, planeNormal);
//    auto ptt = VectorPlaneIntersection(ray.origin, ray.dir, vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
//    auto tt = length( ptt - ray.origin) * 255.0f;
    /** Write result in the output_ranges data array. */
    if (output_ranges->renderInTexture) {
        element.x = (unsigned char)float2uint(t_far * 100.0f,cudaRoundNearest)%255;
        element.y = (unsigned char)float2uint(t_far * 100.0f,cudaRoundNearest)%255;
        surf2Dwrite<uchar4>(element, output_ranges->surface, x * sizeof(uchar4), y);
    } else {
        output_ranges->data[x * output_ranges->dim.y + y] = t_near;
    }
}

extern "C" void integration_range_bbox_wrapper(GPUData<CameraDescriptor>& camera, IntegrationRangeDescriptor *output_ranges,
                                               BBoxDescriptor *bbox) {
    dim3 threads(16, 8);
    /** This create enough blocks to cover the whole texture, may contain threads that does not have pixel's assigned. */
    dim3 blocks((camera.Host()->width + threads.x - 1) / threads.x,
                (camera.Host()->height + threads.y - 1) / threads.y);

    IntegrationRange<<<blocks, threads>>>(camera.Device(), output_ranges, bbox);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}
