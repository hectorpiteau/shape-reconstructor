/*
Author: Hector Piteau (hector.piteau@gmail.com)
PlaneCut.cpp (c) 2023
Desc: Plane Cut
Created:  2023-06-05T21:27:45.699Z
Modified: 2023-06-06T07:41:43.500Z
*/

#include "../view/OverlayPlane.hpp"
#include "../view/SceneObject/SceneObject.hpp"
#include <memory>
#include <utility>
#include "PlaneCut.hpp"
#include "CudaTexture.hpp"
#include "../controllers/Scene/Scene.hpp"
#include "Volume/DenseVolume3D.hpp"

PlaneCut::PlaneCut(Scene *scene, std::shared_ptr<DenseVolume3D> targetVolume) : SceneObject{std::string("PlaneCut"),
                                                                                            SceneObjectTypes::PLANECUT},
                                                                                m_scene(scene),
                                                                                m_dir(PlaneCutDirection::Y),
                                                                                m_pos(vec3(0, 0, 0)),
                                                                                m_targetVolume(std::move(targetVolume)), m_cursorPixel() {
    SetName("PlaneCut");
    /** Create the overlay plane that will be used to display the volume rendering texture on. */
    m_overlay = std::make_shared<OverlayPlane>(
            std::make_shared<ShaderPipeline>("../src/shaders/v_overlay_plane.glsl",
                                             "../src/shaders/f_overlay_plane.glsl"), scene->GetSceneSettings());

    /** Create the cuda texture that will receive the result of the volume rendering process. */
    m_cudaTex = std::make_shared<CudaTexture>(
            scene->GetSceneSettings()->GetViewportWidth(),
            scene->GetSceneSettings()->GetViewportHeight());

    m_cursorPixel.Host()->value = vec4(0.0);
}

void PlaneCut::SetDirection(PlaneCutDirection dir) {
    m_dir = dir;
}

PlaneCutDirection PlaneCut::GetDirection() {
    return m_dir;
}

void PlaneCut::SetPosition(float value) {
    m_pos[m_dir] = value;
}

float PlaneCut::GetPosition() {
    return m_pos[m_dir];
}

vec4 PlaneCut::GetCursorPixelValue(){
    return m_cursorPixel.Host()->value;
}

PlaneCutMode PlaneCut::GetMode(){
    return m_mode;
}

void PlaneCut::Render() {
    std::shared_ptr<Camera> cam = m_scene->GetActiveCam();
    /** Camera desc. TODO: Simplify by allocating it in the Camera Object directly and sharing the descriptor.*/
    double x,y;
    glfwGetCursorPos(m_scene->GetWindow(), &x, &y);
    m_cursorPixel.Host()->loc = ivec2(x,y);
    m_cursorPixel.ToDevice();

    m_cameraDesc.Host()->camExt = cam->GetExtrinsic();
    m_cameraDesc.Host()->camInt = cam->GetIntrinsic();
    m_cameraDesc.Host()->camPos = cam->GetPosition();
    m_cameraDesc.Host()->width = cam->GetResolution().x;
    m_cameraDesc.Host()->height = cam->GetResolution().y;
    m_cameraDesc.ToDevice();

    /** Filling the volume descriptor.  TODO: Same than cameraDesc. */
    m_volumeDesc.Host()->bboxMin = m_targetVolume->GetBboxMin();
    m_volumeDesc.Host()->bboxMax = m_targetVolume->GetBboxMax();
    m_volumeDesc.Host()->worldSize = m_targetVolume->GetBboxMax() - m_targetVolume->GetBboxMin();
    m_volumeDesc.Host()->res = m_targetVolume->GetCudaVolume()->GetResolution();
    m_volumeDesc.Host()->data = m_targetVolume->GetCudaVolume()->GetDevicePtr();
    m_volumeDesc.ToDevice();

    m_planeCutDesc.Host()->axis = m_dir;
    m_planeCutDesc.Host()->pos = m_pos[m_dir];
    m_planeCutDesc.Host()->min = m_targetVolume->GetBboxMin();
    m_planeCutDesc.Host()->max = m_targetVolume->GetBboxMax();
    m_planeCutDesc.Host()->outSurface = m_cudaTex->OpenSurface();
    m_planeCutDesc.Host()->mode = m_mode;
    m_planeCutDesc.ToDevice();

    /** Run kernel on texture. */
//    plane_cut_rendering_wrapper(m_planeCutDesc, m_volumeDesc, m_cameraDesc, m_cursorPixel);
    if(m_s_targetVolume != nullptr){
        sparse_plane_cut_rendering_wrapper(m_planeCutDesc, m_s_targetVolume->GetDescriptor(), m_cameraDesc, m_cursorPixel);
    }

    m_cudaTex->CloseSurface();

    m_overlay->Render(true, m_cudaTex->GetTex());

    m_cursorPixel.ToHost();
}

void PlaneCut::SetMode(PlaneCutMode mode) {
    m_mode = mode;
}

void PlaneCut::SetTargetVolume(std::shared_ptr<Volume3D> vol) {
    if(vol->GetType() == SceneObjectTypes::DENSEVOLUME3D){
        m_targetVolume = std::dynamic_pointer_cast<DenseVolume3D>(vol);
    }else if(vol->GetType() == SceneObjectTypes::SPARSEVOLUME3D){
        m_s_targetVolume = std::dynamic_pointer_cast<SparseVolume3D>(vol);
    }
}
