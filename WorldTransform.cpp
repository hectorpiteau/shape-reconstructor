#include "glm/glm.hpp"
#include "WorldTransform.hpp"
#include "MMath.hpp"

void WorldTransform::SetScale(double scale)
{
    _scale = scale;
}

void WorldTransform::SetRotation(double x, double y, double z)
{
    _rotation.x = x;
    _rotation.y = y;
    _rotation.z = z;
}

void WorldTransform::SetPosition(double x, double y, double z)
{
    _position.x = x;
    _position.y = y;
    _position.z = z;
}

void WorldTransform::Rotate(double x, double y, double z)
{
    _rotation.x += x;
    _rotation.y += y;
    _rotation.z += z;
}

glm::mat4 WorldTransform::GetMatrix()
{
    glm::mat4 scale;
    MMath::InitializeMat4ForScale(scale, _scale, _scale, _scale);

    glm::mat4 rotation;
    MMath::InitializeMat4ForRotation(rotation, _rotation.x, _rotation.y, _rotation.z);

    glm::mat4 translation;
    MMath::InitializeMat4ForScale(translation, _position.x, _position.y, _position.z);

    glm::mat4 world_transform = translation * rotation * scale;

    return world_transform;
}