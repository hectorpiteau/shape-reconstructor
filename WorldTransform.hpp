#ifndef _WORLD_TRANSFORM_H_
#define _WORLD_TRANSFORM_H_
#include "glm/glm.hpp"

class WorldTransform
{
public:
    WorldTransform() {}

    void SetScale(double scale);
    void SetRotation(double x, double y, double z);
    void SetPosition(double x, double y, double z);

    void Rotate(double x, double y, double z);

    glm::mat4 GetMatrix();

private:
    double _scale = 1.0;
    glm::vec3 _rotation = glm::vec3(0, 0, 0);
    glm::vec3 _position = glm::vec3(0, 0, 0);
}

#endif //_WORLD_TRANSFORM_H_