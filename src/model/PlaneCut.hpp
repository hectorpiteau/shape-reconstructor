/*
Author: Hector Piteau (hector.piteau@gmail.com)
PlaneCut.hpp (c) 2023
Desc: Plane cut
Created:  2023-06-05T21:21:40.800Z
Modified: 2023-06-05T23:47:35.584Z
*/
#pragma once

#include "../view/OverlayPlane.hpp"
#include "../view/SceneObject/SceneObject.hpp"
#include "../controllers/Scene/Scene.hpp"
#include "CudaTexture.hpp"
#include "Volume3D.hpp"
#include <glm/glm.hpp>
#include <memory>

using namespace glm;

enum PlaneCutDirection {
    X,Y,Z
};

class PlaneCut : public SceneObject {
private:
    Scene* m_scene;
    /** in-dep */
    /** The overlay to show on screen where the cut-plane get rendered.*/
    std::shared_ptr<OverlayPlane> m_overlay;
    /** The cuda texture used by cuda kernel to write into the result of the cut. */
    std::shared_ptr<CudaTexture> m_cudaTex;

    /** The direction of the cut. following one of the axis.*/
    PlaneCutDirection m_dir;
    /** Ths position of the cut on the axis.*/
    vec3 m_pos;
    /** The target volume to cut and render on the planeCut. */
    std::shared_ptr<Volume3D> m_targetVolume;

    /* Descriptors. */
    GPUData<CameraDescriptor> m_cameraDesc;
    GPUData<PlaneCutDescriptor> m_planeCutDesc;
    GPUData<VolumeDescriptor> m_volumeDesc;
    GPUData<CursorPixel> m_cursorPixel;

public:

    /**
     * Construct a PlaneCut.
     *
     * @param scene : The scene where the cut-plane is assigned to.
     * @param targetVolume : The volume where it will read from.
     */
    explicit PlaneCut(Scene* scene, std::shared_ptr<Volume3D> targetVolume);

    PlaneCut(const PlaneCut&) = delete;
    ~PlaneCut() override = default;

    /**
     * Set the direction of the cut. Cut is axis aligned.
     * @param dir : Either X,Y or Z.
     */
    void SetDirection(PlaneCutDirection dir);

    /**
     * Get the main direction of alignment of the cut-plane.
     * @return Either X,Y or Z.
     */
    PlaneCutDirection GetDirection();

    /**
     * Set the position of the current cut-plane along its axis.
     * @param value : A floating point value used to set the
     * cut-plane position along its axis.
     */
    void SetPosition(float value);

    /**
     * Get the position of the current cut-plane along its axis.
     * @return : The position as a floating point.
     */
    float GetPosition();
    
    void Render() override;

    vec4 GetCursorPixelValue();
};

