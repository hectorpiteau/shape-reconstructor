//
// Created by hpiteau on 14/06/23.
//

#ifndef RT3DRS_INTEGRATION_RANGE_CUH
#define RT3DRS_INTEGRATION_RANGE_CUH
#include "Common.cuh"
#include "GPUData.cuh"

extern "C" void integration_range_bbox_wrapper(GPUData<CameraDescriptor>& camera, IntegrationRangeDescriptor* output_ranges, BBoxDescriptor* bbox);

#endif //RT3DRS_INTEGRATION_RANGE_CUH
