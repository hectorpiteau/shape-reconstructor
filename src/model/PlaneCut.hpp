/*
Author: Hector Piteau (hector.piteau@gmail.com)
PlaneCut.hpp (c) 2023
Desc: Plane cut
Created:  2023-06-05T21:21:40.800Z
Modified: 2023-06-05T23:47:35.584Z
*/
#include <memory>
#include "../view/OverlayPlane.hpp"
#include "CudaTexture.hpp"
#include <glm/glm.hpp>

using namespace glm;

enum PlaneCutDirection {
    X,Y,Z
};

class PlaneCut {
private:
    std::shared_ptr<OverlayPlane> m_overlay;
    std::shared_ptr<CudaTexture> m_cudaTex;
    
    PlaneCutDirection m_dir;

public:

    PlaneCut();
    PlaneCut(const PlaneCut&) = delete;
    ~PlaneCut();

    void SetDirection(PlaneCutDirection dir);
    
    void Render();
};

