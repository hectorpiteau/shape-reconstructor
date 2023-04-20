/*
Author: Hector Piteau (hector.piteau@gmail.com)
RayCasterParams.cuh (c) 2023
Desc: struct RayCasterParams
Created:  2023-04-17T11:39:15.383Z
Modified: 2023-04-17T11:40:13.076Z
*/

#ifndef RAY_CASTER_PARAMS_H
#define RAY_CASTER_PARAMS_H
#include <glm/glm.hpp>

using namespace glm;

struct RayCasterParams {
    /** Camera's intrinsic matrix. */
    mat4 intrinsic;
    /** Camera's extrinsic matrix. */
    mat4 extrinsic;
    /** Camera's world position. */
    vec3 worldPos;
    
    /** Camera's image plane resolution width / height. */
    int width, height;
};

#endif //RAY_CASTER_PARAMS_H

