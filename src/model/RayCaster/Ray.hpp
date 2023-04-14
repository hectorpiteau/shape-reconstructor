/*
Author: Hector Piteau (hector.piteau@gmail.com)
Ray.hpp (c) 2023
Desc: Ray
Created:  2023-04-14T09:51:59.638Z
Modified: 2023-04-14T13:22:44.743Z
*/
#include <glm/glm.hpp>

struct Ray
{
    glm::vec3 origin;
    glm::vec3 dir;
    float tmin, tmax;
};
