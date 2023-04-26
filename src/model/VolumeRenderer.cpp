/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRenderer.cpp (c) 2023
Desc: Implementation for the VolumeRenderer
Created:  2023-04-14T14:45:11.682Z
Modified: 2023-04-26T11:55:28.545Z
*/
#include <memory>
#include <vector>
#include <string> 
#include <glm/glm.hpp>

#include "VolumeRenderer.hpp"

#include "../view/SceneObject/SceneObject.hpp"
#include "../view/OverlayPlane.hpp"
#include "../view/Lines.hpp"
// #include "RayCaster/RayCaster.hpp"
// #include "RayCaster/SingleRayCaster.hpp"
#include "Volume3D.hpp"
#include "Texture2D.hpp"
#include "Camera/Camera.hpp"

#include "../cuda/VolumeRendering.cuh"
#include "../cuda/Projection.cuh"
#include "RayCaster/RayCaster.hpp"

#include "../../include/icons/IconsFontAwesome6.h"

using namespace glm;

VolumeRenderer::VolumeRenderer(Scene* scene)
: SceneObject{std::string("VolumeRenderer"), SceneObjectTypes::VOLUMERENDERER}, m_scene(scene){
    SetName(std::string(ICON_FA_SPINNER " Volume Renderer"));

    m_volume = std::make_shared<Volume3D>(scene, ivec3(128,128,128));
    m_scene->Add(m_volume, true, true);
    m_children.push_back(m_volume);

    // m_outTex = std::make_shared<Texture2D>();
    // m_outPlane = std::make_shared<OverlayPlane>();

    m_camera = scene->GetDefaultCam();
    m_rayCaster = std::make_shared<RayCaster>(m_camera);

    m_renderZoneLines = std::make_shared<Lines>(scene, m_renderingZoneVertices, 4 * 2 * 3);
    m_renderZoneLines->SetColor(vec4(0.0,1.0,0.0,1.0));
}

VolumeRenderer::~VolumeRenderer(){
}

void VolumeRenderer::SetUseDefaultCamera(bool useDefaultCamera){
    m_useDefaultCamera = useDefaultCamera;
}

void VolumeRenderer::ComputeRenderingZone()
{
    vec3* bboxPoints = m_volume->m_bboxPoints;
    
   vec2 ndcMin = vec2(1, 1), ndcMax = vec2(-1,-1);
    for(int i=0; i<8; ++i){
        auto camcoords = WorldToCamera(vec4(bboxPoints[i], 1.0f), m_camera->GetExtrinsic());
        auto ndccoords = CameraToNDC(vec3(camcoords), m_camera->GetIntrinsic());
        ndcMin = min(ndcMin, ndccoords);
        ndcMax = max(ndcMax, ndccoords);
    }

    ndcMin -= vec2(0.02f, 0.02f);
    ndcMax += vec2(0.02f, 0.02f);
    
    m_renderZoneMinNDC = ndcMin;
    m_renderZoneMaxNDC = ndcMax;

    auto p00 = NDCToCamera(vec2(ndcMin.x, ndcMax.y), m_camera->GetIntrinsic()) * -1.0f;
    auto p10 = NDCToCamera(vec2(ndcMax.x, ndcMax.y), m_camera->GetIntrinsic()) * -1.0f;
    auto p20 = NDCToCamera(vec2(ndcMax.x, ndcMin.y), m_camera->GetIntrinsic()) * -1.0f;
    auto p30 = NDCToCamera(vec2(ndcMin.x, ndcMin.y), m_camera->GetIntrinsic()) * -1.0f;
    
    p00 = vec3(CameraToWorld(vec4(p00, 1.0f), m_camera->GetExtrinsic()));
    p10 = vec3(CameraToWorld(vec4(p10, 1.0f), m_camera->GetExtrinsic()));
    p20 = vec3(CameraToWorld(vec4(p20, 1.0f), m_camera->GetExtrinsic()));
    p30 = vec3(CameraToWorld(vec4(p30, 1.0f), m_camera->GetExtrinsic()));
    
    WRITE_VEC3(m_renderingZoneVertices, 0, p00);
    WRITE_VEC3(m_renderingZoneVertices, 3, p10);
    WRITE_VEC3(m_renderingZoneVertices, 6, p10);
    WRITE_VEC3(m_renderingZoneVertices, 9, p20);
    WRITE_VEC3(m_renderingZoneVertices, 12, p20);
    WRITE_VEC3(m_renderingZoneVertices, 15, p30);
    WRITE_VEC3(m_renderingZoneVertices, 18, p30);
    WRITE_VEC3(m_renderingZoneVertices, 21, p00);

    m_renderZoneLines->UpdateVertices(m_renderingZoneVertices);
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

    /** Rendering zone. */
    ComputeRenderingZone();
    m_renderZoneLines->Render();

    for(auto& child : m_children){
        if(child->IsActive()) child->Render();
    }
}


const vec2& VolumeRenderer::GetRenderingZoneMinNDC(){
    return m_renderZoneMinNDC;
}

const vec2& VolumeRenderer::GetRenderingZoneMaxNDC(){
    return m_renderZoneMaxNDC;
}

void VolumeRenderer::SetShowRenderingZone(bool visible){
    m_showRenderingZone = visible;
}

bool VolumeRenderer::ShowRenderingZone(){
    return m_showRenderingZone;
}

std::vector<std::shared_ptr<Camera>> VolumeRenderer::GetAvailableCameras(){
    auto tmp = m_scene->GetAll(SceneObjectTypes::CAMERA);
    std::vector<std::shared_ptr<Camera>> res;
    for(auto x: tmp) res.push_back(std::dynamic_pointer_cast<Camera>(x));
    return res;
}

void VolumeRenderer::SetTargetCamera(std::shared_ptr<Camera> cam){
    m_camera = cam;
}

std::shared_ptr<Camera> VolumeRenderer::GetTargetCamera(){
    return m_camera;
}