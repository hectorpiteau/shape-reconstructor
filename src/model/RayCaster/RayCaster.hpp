/*
Author: Hector Piteau (hector.piteau@gmail.com)
RayCaster.hpp (c) 2023
Desc: Ray caster is used to define rays outgoing from a camera.
Created:  2023-04-14T09:50:13.297Z
Modified: 2023-04-14T12:28:48.345Z
*/
#include <memory>
#include "Ray.hpp"
#include "../Camera/Camera.hpp"

class RayCaster
{
protected:
    std::shared_ptr<Camera> m_camera;
public:
    RayCaster(std::shared_ptr<Camera> camera) : m_camera(camera) {};
    ~RayCaster() {}
    RayCaster(const RayCaster&) = delete;

    void SetCamera(std::shared_ptr<Camera> camera) { m_camera = camera;}

    virtual Ray GetRay(const glm::vec2& pixel) = 0;
};



