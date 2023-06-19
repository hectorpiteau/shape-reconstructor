/*
Author: Hector Piteau (hector.piteau@gmail.com)
RayCaster.hpp (c) 2023
Desc: Ray caster is used to define rays outgoing from a camera.
Created:  2023-04-14T09:50:13.297Z
Modified: 2023-04-26T14:15:22.040Z
*/
#pragma once

#include <memory>
#include <string>
#include <utility>
#include "Ray.h"
#include "../Camera/Camera.hpp"

#include "../../cuda/VolumeRendering.cuh"
#include "../../cuda/Projection.cuh"

#include "../../view/SceneObject/SceneObject.hpp"
#include "../../controllers/Scene/Scene.hpp"

using namespace glm;

/**
 * @brief The RayCaster is the class responsible 
 * for distributing rays along a discrete space. 
 * 
 */
class RayCaster : public SceneObject
{
protected:
    Scene* m_scene;
    /** Camera used to cast rays in the scene. */
    std::shared_ptr<Camera> m_camera;

    /** The rendering zone in the camera view. It determines a 
     * bouding box for casting only useful rays. NDC coordinates. */
    vec2 renderingZoneNDCMin{};
    vec2 renderingZoneNDCMax{};

    /** The rendering zone in the camera view but in pixels,
     * used to know which pixels are rendered, and which are not.
     */
    ivec2 renderingZonePixelMin{};
    ivec2 renderingZonePixelMax{};

    /**
     * The rendering Zone size in Pixels.
     */
    size_t renderingZoneWidth;
    size_t renderingZoneHeight;

    /** Show the rays or not in the view.*/
    bool m_showRays = false;

    Lines* m_rayLines = nullptr;
    float* m_rayLinesVertices = nullptr;

public:
    RayCaster(Scene* scene, std::shared_ptr<Camera> camera)
    : SceneObject{std::string("RayCaster"), SceneObjectTypes::RAYCASTER},
    m_scene(scene), m_camera(std::move(camera)) {
        SetName(std::string("Simple RayCaster"));
        renderingZoneNDCMin = vec2(-1.0, -1.0);
        renderingZoneNDCMax = vec2(1.0, 1.0);

        renderingZonePixelMin = floor(NDCToPixel(renderingZoneNDCMin, m_camera->GetResolution().x, m_camera->GetResolution().y));
        renderingZonePixelMax = ceil(NDCToPixel(renderingZoneNDCMax, m_camera->GetResolution().x, m_camera->GetResolution().y));

        renderingZoneWidth = renderingZonePixelMax.x - renderingZonePixelMin.x;
        renderingZoneHeight = renderingZonePixelMax.y - renderingZonePixelMin.y;
        UpdateRays();
    };

    ~RayCaster() override {
        delete [] m_rayLinesVertices;
        delete m_rayLines;
    }
    
    RayCaster(const RayCaster&) = delete;

    void SetCamera(std::shared_ptr<Camera> camera) { 
        m_camera = std::move(camera);
        UpdateRays();
    }

    ivec2 GetRenderingZoneMinPixel(){
        return renderingZonePixelMin;
    }

    ivec2 GetRenderingZoneMaxPixel(){
        return renderingZonePixelMax;
    }

    void SetRenderingZoneNDC(const vec2& zoneNDCMin, const vec2& zoneNDCMax){
        bool ok = false;
        if(any(notEqual(renderingZoneNDCMin, zoneNDCMin))){
            renderingZoneNDCMin = zoneNDCMin;
            ok = true;
        }
        if(any(notEqual(renderingZoneNDCMax, zoneNDCMax))){
            renderingZoneNDCMax = zoneNDCMax;
            ok = true;
        }
        if(!ok) return;
        
        renderingZonePixelMin = floor(NDCToPixel(zoneNDCMin, m_camera->GetResolution().x, m_camera->GetResolution().y));
        renderingZonePixelMax = ceil(NDCToPixel(zoneNDCMax, m_camera->GetResolution().x, m_camera->GetResolution().y));

        renderingZonePixelMin.x = max(renderingZonePixelMin.x, 0);
        renderingZonePixelMin.y = max(renderingZonePixelMin.y, 0);
//
        renderingZonePixelMax.x = min(renderingZonePixelMax.x, m_camera->GetResolution().x);
        renderingZonePixelMax.y = min(renderingZonePixelMax.y, m_camera->GetResolution().y);

        renderingZoneWidth = renderingZonePixelMax.x - renderingZonePixelMin.x;
        renderingZoneHeight = renderingZonePixelMax.y - renderingZonePixelMin.y;

        UpdateRays();
    }

    [[nodiscard]] size_t GetAmountOfRays() const{
        return renderingZoneWidth * renderingZoneHeight;
    }

    [[nodiscard]] size_t GetRenderZoneWidth() const{
        return renderingZoneWidth;
    }
    
    [[nodiscard]] size_t GetRenderZoneHeight() const{
        return renderingZoneHeight;
    }

    virtual Ray GetRay(const vec2& pixel) {
        Ray tmp{};
        tmp.dir = vec3(1.0);
        tmp.origin = vec3(0.0);
        tmp.tmin = 0.0f;
        tmp.tmax = 1.0f;
        return tmp;
    }

    void UpdateRays(){
        if(!m_showRays) return;
        delete [] m_rayLinesVertices;
        delete  m_rayLines;

        size_t dataLength = 2 * 3 * GetAmountOfRays();
        m_rayLinesVertices = new float[dataLength];
        memset(m_rayLinesVertices, 0, dataLength * sizeof(float));

        size_t index = 0;
        auto origin = m_camera->GetPosition();

        for(int x=renderingZonePixelMin.x; x < renderingZonePixelMax.x; x += 16){
            for(int y=renderingZonePixelMin.y; y < renderingZonePixelMax.y; y += 16){
                WRITE_VEC3(m_rayLinesVertices, index, origin);
                index += 3;

                auto dest = PixelToWorld(vec2(x,y),
                    m_camera->GetIntrinsic(), m_camera->GetExtrinsic(), 
                    m_camera->GetResolution().x,
                    m_camera->GetResolution().y
                );
                auto dir =  dest - origin;
                auto dest2 = dest + dir* 8.0f;

                WRITE_VEC3(m_rayLinesVertices, index, dest2);
                index += 3;
            }
        }

        m_rayLines = new Lines(m_scene, m_rayLinesVertices, dataLength);
    }

    void SetRaysVisible(bool visible){
        m_showRays = visible;
        if(m_showRays) UpdateRays();
    }

    bool AreRaysVisible(){
        return m_showRays;
    }

    void Render(){
        if(m_showRays && m_rayLines != nullptr && m_rayLinesVertices != nullptr){
            m_rayLines->Render();
        }
    }
};



