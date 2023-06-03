/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRenderer.hpp (c) 2023
Desc: description
Created:  2023-04-14T09:48:58.410Z
Modified: 2023-04-26T11:14:57.290Z
*/
#pragma once

#include "RayCaster/RayCaster.hpp"
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
#include "CudaTexture.hpp"

#include "../cuda/Common.cuh"

#include "RayCaster/RayCaster.hpp"
#include "GPUDataStruct/GPUData.hpp"

using namespace glm;

class VolumeRenderer : public SceneObject {
public:
    VolumeRenderer(Scene* scene);
    VolumeRenderer(const VolumeRenderer&) = delete;
    ~VolumeRenderer();

    /**
     * @brief Set the boolean to known if it needs to use the scene active camera or a custom camera.
     * 
     * @param useDefaultCamera : True to use the default camera, false to use a custom camera.
     */
    void SetUseDefaultCamera(bool useDefaultCamera);

    /**
     * @brief Get the Rendering Zone's minimum Normalized Device Coordinates.
     * 
     * @return const vec2& : A constant ref to a vec2 containing coordinates in [-1, 1].
     */
    const vec2& GetRenderingZoneMinNDC();
    
    /**
     * @brief Get the Rendering Zone's maximum Normalized Device Coordinates.
     * 
     * @return const vec2& : A constant ref to a vec2 containing coordinates in [-1, 1].
     */
    const vec2& GetRenderingZoneMaxNDC();

    /**
     * @brief Set the Show Rendering Zone boolean.
     * 
     * @param visible : True to display the rendering zone on the target camera. False to hide it.
     */
    void SetShowRenderingZone(bool visible);

    /**
     * @brief Get the showRenderingZone boolean.
     * 
     * @return True to display the rendering zone on the target camera. 
     * @return False to hide it.
     */
    bool GetShowRenderingZone();

    /**
     * @brief Compute the zone based on both the camera parameters and 
     * the volume configuration.
     */
    void ComputeRenderingZone();
    std::shared_ptr<Lines> m_renderZoneLines;
    float m_renderingZoneVertices[4 * 2 * 3] = {0.0f};

    /**
     * @brief Get the Available Cameras in the scene.
     * TODO: Move closer to the view.
     * @return std::vector<std::shared_ptr<Camera>> : A list of available cameras.
     */
    std::vector<std::shared_ptr<Camera>> GetAvailableCameras();

    /**
     * @brief Set the Target Camera to use if not using the default scene's 
     * camera.
     * 
     * @param cam : A shared ptr to the new target camera. 
     */
    void SetTargetCamera(std::shared_ptr<Camera> cam);

    /**
     * @brief Get the Target Camera to be used by the renderer.
     * 
     * @return std::shared_ptr<Camera> 
     */
    std::shared_ptr<Camera> GetTargetCamera();

    /**
     * @brief Render the VolumeRenderer in the Scene.
     * It includes potential visual cues of the renderer, but
     * also the result of the volume renderer itself.
     */
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
    std::shared_ptr<RayCaster> m_rayCaster;
    std::shared_ptr<CudaTexture> m_cudaTex;

    /* Descriptors. */
    GPUData<CameraDescriptor> m_cameraDesc;
    GPUData<VolumeDescriptor> m_volumeDesc;
    GPUData<RayCasterDescriptor> m_raycasterDesc;
    
    RayCasterParams m_params;
    // void RunKernel(cudaArray *image_array, uint width, uint height);
};