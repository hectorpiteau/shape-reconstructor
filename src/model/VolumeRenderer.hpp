/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRenderer.hpp (c) 2023
Desc: description
Created:  2023-04-14T09:48:58.410Z
Modified: 2023-04-17T09:43:58.230Z
*/
#pragma once

// #include "RayCaster/RayCaster.hpp"
#include "Texture2D.hpp"
#include "Volume3D.hpp"
#include "../view/SceneObject/SceneObject.hpp"
#include "../view/OverlayPlane.hpp"
#include "../controllers/Scene/Scene.hpp"

class VolumeRenderer : public SceneObject {
public:
    VolumeRenderer(Scene* scene, std::shared_ptr<Volume3D> volume);
    VolumeRenderer(const VolumeRenderer&) = delete;
    ~VolumeRenderer();

    void SetCamera(std::shared_ptr<Camera> camera);

    void SetUseDefaultCamera(bool useDefaultCamera);

    void Render();

private:
    bool m_useDefaultCamera;

    /** out dep. */
    Scene* m_scene;
    std::shared_ptr<Volume3D> m_volume;
    std::shared_ptr<Camera> m_camera;

    /** in dep. */
    std::shared_ptr<OverlayPlane> m_outPlane;
    // std::shared_ptr<RayCaster> m_rayCaster;
    std::shared_ptr<Texture2D> m_outTex;
};