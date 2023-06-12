/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRendererInteractor.hpp (c) 2023
Desc: VolumeRendererInteractor
Created:  2023-04-21T13:35:05.404Z
Modified: 2023-04-26T09:20:04.270Z
*/
#pragma once
#include <memory>
#include <vector>

#include "../model/VolumeRenderer.hpp"
#include "../model/Camera/Camera.hpp"
#include "../controllers/Scene/Scene.hpp"

class VolumeRendererInteractor{
private:
    std::shared_ptr<VolumeRenderer> m_volumeRenderer;
    Scene* m_scene;

    std::shared_ptr<Camera> m_camera;
    std::vector<std::shared_ptr<Camera>> m_availableCameras;
public:

    VolumeRendererInteractor(Scene* scene);
    VolumeRendererInteractor(const VolumeRendererInteractor&) = delete;
    ~VolumeRendererInteractor();
    
    const vec2& GetRenderingZoneMinNDC();
    const vec2& GetRenderingZoneMaxNDC();
    
    const vec2 GetRenderingZoneMinPixel();
    const vec2 GetRenderingZoneMaxPixel();
    
    std::vector<std::shared_ptr<Camera>>& GetAvailableCameras();
    
    void SetTargetCamera(std::shared_ptr<Camera> cam);
    // std::shared_ptr<Camera> GetTargetCamera();

    void SetIsRenderingZoneVisible(bool visible);
    bool IsRenderingZoneVisible();


    bool IsRendering();
    void SetIsRendering(bool value);

    void SetCurrentVolumeRenderer(std::shared_ptr<VolumeRenderer> volumeRenderer);
};