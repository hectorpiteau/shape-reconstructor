/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRendererInteractor.hpp (c) 2023
Desc: VolumeRendererInteractor
Created:  2023-04-21T13:35:05.404Z
Modified: 2023-04-26T11:30:43.827Z
*/
#include <memory>
#include <vector>
#include "../model/VolumeRenderer.hpp"
#include "../cuda/Projection.cuh"
#include "../controllers/Scene/Scene.hpp"
#include "../model/Camera/Camera.hpp"
#include "VolumeRendererInteractor.hpp"

VolumeRendererInteractor::VolumeRendererInteractor(Scene* scene) : m_scene(scene) {
    m_camera = scene->GetDefaultCam();
}

VolumeRendererInteractor::~VolumeRendererInteractor() {}

void VolumeRendererInteractor::SetCurrentVolumeRenderer(std::shared_ptr<VolumeRenderer> volumeRenderer)
{
    m_volumeRenderer = volumeRenderer;
    m_availableCameras = m_volumeRenderer->GetAvailableCameras(); 
}

const vec2 VolumeRendererInteractor::GetRenderingZoneMinPixel(){
    return NDCToPixel(
        m_volumeRenderer->GetRenderingZoneMinNDC(),
        m_scene->GetSceneSettings()->GetViewportWidth(), 
        m_scene->GetSceneSettings()->GetViewportHeight()
        );
}

const vec2 VolumeRendererInteractor::GetRenderingZoneMaxPixel(){
    return NDCToPixel(
        m_volumeRenderer->GetRenderingZoneMaxNDC(),
        m_scene->GetSceneSettings()->GetViewportWidth(), 
        m_scene->GetSceneSettings()->GetViewportHeight()
        );
}

std::vector<std::shared_ptr<Camera>>& VolumeRendererInteractor::GetAvailableCameras(){
    return m_availableCameras;
}

void VolumeRendererInteractor::SetTargetCamera(std::shared_ptr<Camera> cam){
    m_volumeRenderer->SetTargetCamera(cam);
}

const vec2& VolumeRendererInteractor::GetRenderingZoneMinNDC(){
    return m_volumeRenderer->GetRenderingZoneMinNDC();
}

const vec2& VolumeRendererInteractor::GetRenderingZoneMaxNDC(){
    return m_volumeRenderer->GetRenderingZoneMaxNDC();
}

void VolumeRendererInteractor::SetIsRenderingZoneVisible(bool visible){
    m_volumeRenderer->SetShowRenderingZone(visible);
}

bool VolumeRendererInteractor::IsRenderingZoneVisible(){
    m_volumeRenderer->ShowRenderingZone();
}