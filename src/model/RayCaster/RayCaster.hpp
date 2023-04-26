/*
Author: Hector Piteau (hector.piteau@gmail.com)
RayCaster.hpp (c) 2023
Desc: Ray caster is used to define rays outgoing from a camera.
Created:  2023-04-14T09:50:13.297Z
Modified: 2023-04-26T12:20:42.525Z
*/
#pragma once

#include <memory>
#include <string>
#include "Ray.h"
#include "../Camera/Camera.hpp"

#include "../../cuda/VolumeRendering.cuh"
#include "../../cuda/Projection.cuh"

#include "../../view/SceneObject/SceneObject.hpp"


using namespace glm;

/**
 * @brief The RayCaster is the class responsible 
 * for distributing rays along a discrete space. 
 * 
 */
class RayCaster : public SceneObject
{
protected:
    /** Camera used to cast rays in the scene. */
    std::shared_ptr<Camera> m_camera;

    /** The rendering zone in the camera view. It determines a 
     * bouding box for casting only useful rays. NDC coordinates. */
    vec2 renderingZoneNDCMin;
    vec2 renderingZoneNDCMax;

    /** The rendering zone in the camera view but in pixels,
     * used to know which pixels are rendered, and which are not.
     */
    ivec2 renderingZonePixelMin;
    ivec2 renderingZonePixelMax;

    /**
     * The rendering Zone size in Pixels.
     */
    size_t renderingZoneWidth;
    size_t renderingZoneHeight;

    /** Show the rays or not in the view.*/
    bool m_showRays = false;

    std::shared_ptr<Lines> m_rayLines;
    float* m_rayLinesVertices = nullptr;

public:
    RayCaster(std::shared_ptr<Camera> camera)   
    : SceneObject{std::string("RayCaster"), SceneObjectTypes::RAYCASTER},
    m_camera(camera) {
        SetName(std::string("Simple RayCaster"));
        renderingZoneNDCMin = vec2(-1.0, -1.0);
        renderingZoneNDCMax = vec2(1.0, 1.0);

        renderingZonePixelMin = floor(NDCToPixel(renderingZoneNDCMin, m_camera->GetResolution().x, m_camera->GetResolution().y));
        renderingZonePixelMax = ceil(NDCToPixel(renderingZoneNDCMax, m_camera->GetResolution().x, m_camera->GetResolution().y));

        renderingZoneWidth = renderingZonePixelMax.x - renderingZonePixelMin.x;
        renderingZoneHeight = renderingZonePixelMax.y - renderingZonePixelMin.y;
    };

    ~RayCaster() {
        if(m_rayLinesVertices != nullptr) delete [] m_rayLinesVertices;
    }
    
    RayCaster(const RayCaster&) = delete;

    void SetCamera(std::shared_ptr<Camera> camera) { m_camera = camera;}

    void SetRenderingZoneNDC(const vec2& zoneNDCMin, const vec2& zoneNDCMax){
        renderingZoneNDCMin = zoneNDCMin;
        renderingZoneNDCMax = zoneNDCMax;
        
        renderingZonePixelMin = floor(NDCToPixel(zoneNDCMin, m_camera->GetResolution().x, m_camera->GetResolution().y));
        renderingZonePixelMax = ceil(NDCToPixel(zoneNDCMax, m_camera->GetResolution().x, m_camera->GetResolution().y));
    }

    size_t GetAmountOfRays(){
        return renderingZoneWidth * renderingZoneHeight;
    }

    virtual Ray GetRay(const vec2& pixel) {
        Ray tmp;
        tmp.dir = vec3(1.0);
        tmp.origin = vec3(0.0);
        tmp.tmin = 0.0f;
        tmp.tmax = 1.0f;
        return tmp;
    }

    void UpdateRays(){
        if(m_rayLinesVertices != nullptr) delete [] m_rayLinesVertices;
        
        m_rayLinesVertices = new float[2 * 3 * GetAmountOfRays()];

        size_t index = 0;
        auto origin = m_camera->GetPosition();
        for(int x=renderingZonePixelMin.x; x < renderingZonePixelMax.x; ++x){
            for(int y=renderingZonePixelMin.y; y < renderingZonePixelMax.y; ++y){
                WRITE_VEC3(m_rayLinesVertices, index, origin);
                index += 3;
                auto dest = PixelToWorld(vec2(x,y),
                m_camera->GetIntrinsic(), m_camera->GetExtrinsic(), 
                m_camera->GetResolution().x,
                m_camera->GetResolution().y
                );
                WRITE_VEC3(m_rayLinesVertices, index, dest);
                index += 3;
            }
        }
        
    }

    void SetRaysVisible(bool visible){
        m_showRays = visible;
    }

    bool AreRaysVisible(){
        return m_showRays;
    }

    void Render(){
        if(m_showRays) m_rayLines->Render();
    }
};



