/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRenderer.cpp (c) 2023
Desc: Implementation for the VolumeRenderer
Created:  2023-04-14T14:45:11.682Z
Modified: 2023-04-17T12:22:10.581Z
*/
#include <memory>
#include <string> 

#include "VolumeRenderer.hpp"

#include "../view/SceneObject/SceneObject.hpp"
#include "../view/OverlayPlane.hpp"
// #include "RayCaster/RayCaster.hpp"
// #include "RayCaster/SingleRayCaster.hpp"
#include "Volume3D.hpp"
#include "Texture2D.hpp"

#include "../cuda/VolumeRendering.cuh"

#include "../../include/icons/IconsFontAwesome6.h"


VolumeRenderer::VolumeRenderer(Scene* scene, std::shared_ptr<Volume3D> volume)
: SceneObject{std::string("VolumeRenderer"), SceneObjectTypes::VOLUMERENDERER}, m_scene(scene), m_volume(volume){
    SetName(std::string(ICON_FA_SPINNER " Volume Renderer"));
    m_outPlane = std::make_shared<OverlayPlane>();
    // m_rayCaster = std::make_shared<SingleRayCaster>();
    m_outTex = std::make_shared<Texture2D>();

    // m_children.push_back(m_rayCaster);
    m_children.push_back(m_volume);
    // m_children.push_back(m_outTex);
}

VolumeRenderer::~VolumeRenderer(){
}

void VolumeRenderer::SetCamera(std::shared_ptr<Camera> camera){
    m_camera = camera;
}

void VolumeRenderer::SetUseDefaultCamera(bool useDefaultCamera){
    m_useDefaultCamera = useDefaultCamera;
}

void VolumeRenderer::Render(){

    /** Render volume using the raycaster. */
    // volume_rendering_wrapper(
    //     m_rayCaster,
    //     m_volume,
    //     m_outTex,
    //     (m_useDefaultCamera || m_camera == nullptr)  ? m_scene->GetActiveCam()->GetResolution().x : m_camera->GetResolution().x,
    //     (m_useDefaultCamera || m_camera == nullptr)  ? m_scene->GetActiveCam()->GetResolution().y : m_camera->GetResolution().y
    // ); 

    for(auto& child : m_children){
        if(child->IsActive()) child->Render();
    }
}
