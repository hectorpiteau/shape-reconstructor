/*
Author: Hector Piteau (hector.piteau@gmail.com)
PlaneCut.cpp (c) 2023
Desc: Plane Cut
Created:  2023-06-05T21:27:45.699Z
Modified: 2023-06-06T07:41:43.500Z
*/

#include <memory>
#include "PlaneCut.hpp"
#include "../view/OverlayPlane.hpp"
#include "CudaTexture.hpp"
#include "../controllers/Scene/Scene.hpp"

PlaneCut::PlaneCut(std::shared_ptr<Scene> scene){
    /** Create the overlay plane that will be used to display the volume rendering texture on. */
    m_overlay = std::make_shared<OverlayPlane>(
        std::make_shared<ShaderPipeline>("../src/shaders/v_overlay_plane.glsl", "../src/shaders/f_overlay_plane.glsl")
    );

    /** Create the cuda texture that will receive the result of the volume rendering process. */
    m_cudaTex = std::make_shared<CudaTexture>(
        scene->GetSceneSettings()->GetViewportWidth(),
        scene->GetSceneSettings()->GetViewportHeight()
    );
}

PlaneCut::~PlaneCut(){
    
}

void PlaneCut::SetDirection(PlaneCutDirection dir){
    m_dir = dir;
}
    
void PlaneCut::Render(){
    m_cudaTex->RunCUDAPlaneCut();
}