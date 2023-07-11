/*
Author: Hector Piteau (hector.piteau@gmail.com)
Volume.cuh (c) 2023
Desc: Volume rendering algorithms.
Created:  2023-04-13T12:33:22.433Z
Modified: 2023-04-25T12:53:31.894Z
*/

#ifndef VOLUME_H
#define VOLUME_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <glm/glm.hpp>

#include "../model/RayCaster/Ray.h"
#include "RayCasterParams.cuh"
#include "Common.cuh"
#include "GPUData.cuh"

extern "C" void volume_resize_double_wrapper(cell* source_volume, cell* target_volume, const ivec3& source_res, const ivec3& target_res);


#endif // VOLUME_H