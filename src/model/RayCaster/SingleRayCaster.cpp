#include <memory>
#include <glm/glm.hpp>
#include "Ray.hpp"
#include "RayCaster.hpp"
#include "SingleRayCaster.hpp"
#include "../Camera/Camera.hpp"
#include "../../utils/Projection.hpp"

using namespace glm;

SingleRayCaster::SingleRayCaster(std::shared_ptr<Camera> camera) : RayCaster{camera} {

}

SingleRayCaster::~SingleRayCaster(){

}

Ray SingleRayCaster::GetRay(const vec2& pixel){

    vec3 dir = vec3(0.0);
    /** Compute ray from camera to pixel. Undistort. */
    dir = Projection::PixelToWorld(pixel, m_camera->GetIntrinsic(), m_camera->);

    Ray ray = {
        .origin = m_camera->GetPosition(),
        .dir = dir,
        .tmin = 0,
        .tmax = 1
    };

    return ray;
}