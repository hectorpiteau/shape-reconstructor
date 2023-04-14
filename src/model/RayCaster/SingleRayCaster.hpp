/*
Author: Hector Piteau (hector.piteau@gmail.com)
SingleRayCaster.hpp (c) 2023
Desc: Cast one ray per pixel.
Created:  2023-04-14T11:30:44.487Z
Modified: 2023-04-14T13:20:22.077Z
*/
#include "Ray.hpp"
#include "RayCaster.hpp"

class SingleRayCaster : public RayCaster
{
private:
    
public:
    SingleRayCaster(std::shared_ptr<Camera> camera);
    ~SingleRayCaster();

    Ray GetRay(const glm::vec2& pixel);
};