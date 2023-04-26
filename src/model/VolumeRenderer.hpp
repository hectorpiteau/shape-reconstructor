/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRenderer.hpp (c) 2023
Desc: description
Created:  2023-04-14T09:48:58.410Z
Modified: 2023-04-26T09:37:54.284Z
*/
#pragma once

// #include "RayCaster/RayCaster.hpp"
#include <memory>
#include <vector>
#include <glm/glm.hpp>

#include "Texture2D.hpp"
#include "Volume3D.hpp"
#include "Camera/Camera.hpp"
#include "../view/SceneObject/SceneObject.hpp"
#include "../view/OverlayPlane.hpp"
#include "../controllers/Scene/Scene.hpp"
#include "../view/Lines.hpp"

using namespace glm;

class VolumeRenderer : public SceneObject {
public:
    VolumeRenderer(Scene* scene);
    VolumeRenderer(const VolumeRenderer&) = delete;
    ~VolumeRenderer();

    void SetCamera(std::shared_ptr<Camera> camera);

    void SetUseDefaultCamera(bool useDefaultCamera);

    const vec2& GetRenderingZoneMinNDC();
    const vec2& GetRenderingZoneMaxNDC();

    void SetShowRenderingZone(bool visible);
    bool ShowRenderingZone();

    void ComputeRenderingZone();
    std::shared_ptr<Lines> m_renderZoneLines;
    float m_renderingZoneVertices[4 * 2 * 3] = {0.0f};

    std::vector<std::shared_ptr<Camera>> GetAvailableCameras();
    void SetTargetCamera(std::shared_ptr<Camera> cam);
    std::shared_ptr<Camera> GetTargetCamera();

    void Render();

    size_t amountOfRays = 0;

    vec2 m_renderZoneMinNDC;
    vec2 m_renderZoneMaxNDC;

private:
    bool m_useDefaultCamera = true;
    bool m_showRenderingZone = true;

    /** out dep. */
    Scene* m_scene;
    std::shared_ptr<Volume3D> m_volume;
    std::shared_ptr<Camera> m_camera;

    /** in dep. */
    std::shared_ptr<OverlayPlane> m_outPlane;
    // std::shared_ptr<RayCaster> m_rayCaster;
    std::shared_ptr<Texture2D> m_outTex;
};