/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRenderer.cpp (c) 2023
Desc: Implementation for the VolumeRenderer
Created:  2023-04-14T14:45:11.682Z
Modified: 2023-04-26T13:51:29.197Z
*/
#include <memory>
#include <utility>
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
#include "CudaTexture.hpp"

#include "Camera/Camera.hpp"

#include "../cuda/VolumeRendering.cuh"
#include "../cuda/Projection.cuh"
#include "../cuda/Common.cuh"

#include "RayCaster/RayCaster.hpp"
#include "../cuda/GPUData.cuh"
#include "../../include/icons/IconsFontAwesome6.h"
#include "../utils/SceneGlobalVariables.hpp"

using namespace glm;

VolumeRenderer::VolumeRenderer(Scene *scene, const ivec3& res)
        : SceneObject{std::string("VolumeRenderer"), SceneObjectTypes::VOLUMERENDERER}, m_scene(scene) {
    SetName(std::string(ICON_FA_SPINNER " Volume Renderer"));

    m_volume = std::make_shared<Volume3D>(scene, res);
    m_scene->Add(m_volume, true, true);
    m_children.push_back(m_volume);

    m_camera = scene->GetDefaultCam();
    m_rayCaster = std::make_shared<RayCaster>(m_scene, m_camera);
    m_scene->Add(m_rayCaster, true, true);
    m_children.push_back(m_rayCaster);

    m_renderZoneLines = std::make_shared<Lines>(scene, m_renderingZoneVertices, 4 * 2 * 3);
    m_renderZoneLines->SetColor(vec4(0.0, 1.0, 0.0, 1.0));

    /** Create the overlay plane that will be used to display the volume rendering texture on. */
    m_outPlane = std::make_shared<OverlayPlane>(
            std::make_shared<ShaderPipeline>("../src/shaders/v_overlay_plane.glsl",
                                             "../src/shaders/f_overlay_plane.glsl"),
            m_scene->GetSceneSettings()
    );

    /** Create the cuda texture that will receive the result of the volume rendering process. */
    m_cudaTex = std::make_shared<CudaTexture>(
            m_scene->GetSceneSettings()->GetViewportWidth(),
            m_scene->GetSceneSettings()->GetViewportHeight()
    );
}

void VolumeRenderer::SetUseDefaultCamera(bool useDefaultCamera) {
    m_useDefaultCamera = useDefaultCamera;
    m_camera = m_scene->GetDefaultCam();
    m_rayCaster->SetCamera(m_camera);
}

bool VolumeRenderer::IsRendering() const {
    return m_isRendering;
}

void VolumeRenderer::SetIsRendering(bool value) {
    m_isRendering = value;
}

void VolumeRenderer::ComputeRenderingZone() {
    vec3 *bboxPoints = m_volume->m_bboxPoints;

    vec2 ndcMin = vec2(1, 1), ndcMax = vec2(-1, -1);
    for (int i = 0; i < 8; ++i) {
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
    m_rayCaster->SetRenderingZoneNDC(m_renderZoneMinNDC, m_renderZoneMaxNDC);
}

void VolumeRenderer::Render() {
    m_isRendering = m_scene->GetSceneSettings()->GetVariable(SceneGlobalVariables::VOLUME_RENDERING);
    std::shared_ptr<Camera> cam = (m_useDefaultCamera || m_camera == nullptr) ? m_scene->GetActiveCam() : m_camera;

    ComputeRenderingZone();



    /** Render volume using the raycaster. */
    if (m_isRendering) {
        cam->UpdateGPUDescriptor();
//        m_cameraDesc.Host()->camExt = cam->GetExtrinsic();
//        m_cameraDesc.Host()->camInt = cam->GetIntrinsic();
//        m_cameraDesc.Host()->camPos = cam->GetPosition();
//        m_cameraDesc.Host()->width = cam->GetResolution().x;
//        m_cameraDesc.Host()->height = cam->GetResolution().y;
//        m_cameraDesc.ToDevice();

        m_raycasterDesc.Host()->minPixelX = m_rayCaster->GetRenderingZoneMinPixel().x;
        m_raycasterDesc.Host()->minPixelY = m_rayCaster->GetRenderingZoneMinPixel().y;
        m_raycasterDesc.Host()->maxPixelX = m_rayCaster->GetRenderingZoneMaxPixel().x;
        m_raycasterDesc.Host()->maxPixelY = m_rayCaster->GetRenderingZoneMaxPixel().y;
        m_raycasterDesc.Host()->renderAllPixels = false;
        m_raycasterDesc.Host()->surface = m_cudaTex->OpenSurface();
        m_raycasterDesc.ToDevice();

//        m_volumeDesc.Host()->bboxMin = m_volume->GetBboxMin();
//        m_volumeDesc.Host()->bboxMax = m_volume->GetBboxMax();
//        m_volumeDesc.Host()->worldSize = m_volume->GetBboxMax() - m_volume->GetBboxMin();
//        m_volumeDesc.Host()->res = m_volume->GetCudaVolume()->GetResolution();
//        m_volumeDesc.Host()->data = m_volume->GetCudaVolume()->GetDevicePtr();
//        m_volumeDesc.ToDevice();

        volume_rendering_wrapper(m_raycasterDesc, cam->GetGPUData(), m_volume->GetGPUData());
        m_cudaTex->CloseSurface();
    }

    // /** Rendering zone. */

    if (m_showRenderingZone) {
        m_renderZoneLines->Render();
    }


    for (auto &child: m_children) {
        if (child->IsActive()) child->Render();
    }

    if (m_isRendering) {
        m_outPlane->Render(true, m_cudaTex->GetTex());
    }
}


GPUData<RayCasterDescriptor>& VolumeRenderer::GetRayCasterGPUData() {
    m_raycasterDesc.Host()->minPixelX = m_rayCaster->GetRenderingZoneMinPixel().x;
    m_raycasterDesc.Host()->minPixelY = m_rayCaster->GetRenderingZoneMinPixel().y;
    m_raycasterDesc.Host()->maxPixelX = m_rayCaster->GetRenderingZoneMaxPixel().x;
    m_raycasterDesc.Host()->maxPixelY = m_rayCaster->GetRenderingZoneMaxPixel().y;
    m_raycasterDesc.Host()->renderAllPixels = true;
    return m_raycasterDesc;
}

void VolumeRenderer::UpdateGPUDescriptors(){
    m_volumeDesc.Host()->bboxMin = m_volume->GetBboxMin();
    m_volumeDesc.Host()->bboxMax = m_volume->GetBboxMax();
    m_volumeDesc.Host()->worldSize = m_volume->GetBboxMax() - m_volume->GetBboxMin();
    m_volumeDesc.Host()->res = m_volume->GetCudaVolume()->GetResolution();
    m_volumeDesc.Host()->data = m_volume->GetCudaVolume()->GetDevicePtr();
    m_volumeDesc.ToDevice();
}

GPUData<VolumeDescriptor>& VolumeRenderer::GetVolumeGPUData() {
    return m_volumeDesc;
}

const vec2 &VolumeRenderer::GetRenderingZoneMinNDC() const {
    return m_renderZoneMinNDC;
}

const vec2 &VolumeRenderer::GetRenderingZoneMaxNDC() const {
    return m_renderZoneMaxNDC;
}

void VolumeRenderer::SetShowRenderingZone(bool visible) {
    m_showRenderingZone = visible;
}

bool VolumeRenderer::GetShowRenderingZone() const {
    return m_showRenderingZone;
}

std::vector<std::shared_ptr<Camera>> VolumeRenderer::GetAvailableCameras() {
    auto tmp = m_scene->GetAll(SceneObjectTypes::CAMERA);
    std::vector<std::shared_ptr<Camera>> res;
    /** Pre-allocate room for all the cameras. */
    res.reserve(tmp.size());
    for (const auto &x: tmp) res.push_back(std::dynamic_pointer_cast<Camera>(x));
    return res;
}

void VolumeRenderer::SetTargetCamera(std::shared_ptr<Camera> cam) {
    m_camera = std::move(cam);
    m_rayCaster->SetCamera(m_camera);
}

std::shared_ptr<Camera> VolumeRenderer::GetTargetCamera() {
    return m_camera;
}

std::shared_ptr<Volume3D> VolumeRenderer::GetVolume3D() {
    return m_volume;
}
