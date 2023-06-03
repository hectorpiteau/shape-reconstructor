#include <memory>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include "../../include/icons/IconsFontAwesome6.h"
#include "Volume3D.hpp"

#include "../view/Lines.hpp"
#include "../view/Renderable/Renderable.hpp"
#include "../view/Wireframe/Wireframe.hpp"
#include "../view/SceneObject/SceneObject.hpp"

#include "../controllers/Scene/Scene.hpp"

#include "../cuda/Projection.cuh"
#include "../cuda/CudaLinearVolume3D.cuh"

using namespace glm;

Volume3D::Volume3D(Scene *scene, ivec3 res) : SceneObject{std::string("VOLUME3D"), SceneObjectTypes::VOLUME3D}, m_scene(scene), m_res(res)
{
    SetName(std::string(ICON_FA_CUBES " Volume 3D"));
    m_lines = std::make_shared<Lines>(scene, m_wireframeVertices, 12 * 2 * 3);
    m_lines->SetActive(true);
    ComputeBBoxPoints();
    m_cudaVolume = std::make_shared<CudaLinearVolume3D>(vec3(100, 100, 100));
    m_cudaVolume->InitStub();
    m_cudaVolume->ToGPU();
}

void Volume3D::SetBBoxMin(const vec3 &bboxMin)
{
    m_bboxMin = bboxMin;
    ComputeBBoxPoints();
    m_lines->UpdateVertices(m_wireframeVertices);
}

void Volume3D::SetBBoxMax(const vec3 &bboxMax)
{
    m_bboxMax = bboxMax;
    ComputeBBoxPoints();
    m_lines->UpdateVertices(m_wireframeVertices);
}

void Volume3D::InitializeVolume()
{
    
}

const ivec3 &Volume3D::GetResolution()
{
    return m_res;
}

void Volume3D::ComputeBBoxPoints()
{
    m_bboxPoints[0] = m_bboxMin;
    

    m_bboxPoints[1] = m_bboxMin;
    m_bboxPoints[1].x = m_bboxMax.x;

    m_bboxPoints[2] = m_bboxMin;
    m_bboxPoints[2].y = m_bboxMax.y;

    m_bboxPoints[3] = m_bboxMax;
    m_bboxPoints[3].z = m_bboxMin.z;

    m_bboxPoints[4] = m_bboxMin;
    m_bboxPoints[4].z = m_bboxMax.z;

    m_bboxPoints[5] = m_bboxMax;
    m_bboxPoints[5].y = m_bboxMin.y;

    m_bboxPoints[6] = m_bboxMax;
    m_bboxPoints[6].x = m_bboxMin.x;

    m_bboxPoints[7] = m_bboxMax;

    WRITE_VEC3(m_wireframeVertices, 0, m_bboxPoints[0]); //front face, toward negative z
    WRITE_VEC3(m_wireframeVertices, 3, m_bboxPoints[1]);

    WRITE_VEC3(m_wireframeVertices, 6, m_bboxPoints[1]);
    WRITE_VEC3(m_wireframeVertices, 9, m_bboxPoints[3]);

    WRITE_VEC3(m_wireframeVertices, 12, m_bboxPoints[3]);
    WRITE_VEC3(m_wireframeVertices, 15, m_bboxPoints[2]);
    
    WRITE_VEC3(m_wireframeVertices, 18, m_bboxPoints[2]);
    WRITE_VEC3(m_wireframeVertices, 21, m_bboxPoints[0]);
    
    WRITE_VEC3(m_wireframeVertices, 24, m_bboxPoints[4]); //back face, toward positive z
    WRITE_VEC3(m_wireframeVertices, 27, m_bboxPoints[5]);

    WRITE_VEC3(m_wireframeVertices, 30, m_bboxPoints[5]);
    WRITE_VEC3(m_wireframeVertices, 33, m_bboxPoints[7]);
    
    WRITE_VEC3(m_wireframeVertices, 36, m_bboxPoints[7]);
    WRITE_VEC3(m_wireframeVertices, 39, m_bboxPoints[6]);

    WRITE_VEC3(m_wireframeVertices, 42, m_bboxPoints[6]);
    WRITE_VEC3(m_wireframeVertices, 45, m_bboxPoints[4]);

    WRITE_VEC3(m_wireframeVertices, 48, m_bboxPoints[0]); //connecting two faces
    WRITE_VEC3(m_wireframeVertices, 51, m_bboxPoints[4]); 
    WRITE_VEC3(m_wireframeVertices, 54, m_bboxPoints[1]);
    WRITE_VEC3(m_wireframeVertices, 57, m_bboxPoints[5]);
    WRITE_VEC3(m_wireframeVertices, 60, m_bboxPoints[2]);
    WRITE_VEC3(m_wireframeVertices, 63, m_bboxPoints[6]);
    WRITE_VEC3(m_wireframeVertices, 66, m_bboxPoints[3]);
    WRITE_VEC3(m_wireframeVertices, 69, m_bboxPoints[7]);
}

// void Volume3D::ComputeRenderingZone()
// {
//    vec2 ndcMin = vec2(1, 1), ndcMax = vec2(-1,-1);
//     for(int i=0; i<8; ++i){
//         auto camcoords = WorldToCamera(vec4(m_bboxPoints[i], 1.0f), m_scene->GetActiveCam()->GetExtrinsic());
//         auto ndccoords = CameraToNDC(vec3(camcoords), m_scene->GetActiveCam()->GetIntrinsic());
//         ndcMin = min(ndcMin, ndccoords);
//         ndcMax = max(ndcMax, ndccoords);
//     }

//     ndcMin -= vec2(0.02f, 0.02f);
//     ndcMax += vec2(0.02f, 0.02f);
    
//     m_renderZoneMinNDC = ndcMin;
//     m_renderZoneMaxNDC = ndcMax;

//     auto p00 = NDCToCamera(vec2(ndcMin.x, ndcMax.y), m_scene->GetActiveCam()->GetIntrinsic()) * -1.0f;
//     auto p10 = NDCToCamera(vec2(ndcMax.x, ndcMax.y), m_scene->GetActiveCam()->GetIntrinsic()) * -1.0f;
//     auto p20 = NDCToCamera(vec2(ndcMax.x, ndcMin.y), m_scene->GetActiveCam()->GetIntrinsic()) * -1.0f;
//     auto p30 = NDCToCamera(vec2(ndcMin.x, ndcMin.y), m_scene->GetActiveCam()->GetIntrinsic()) * -1.0f;
    
//     p00 = vec3(CameraToWorld(vec4(p00, 1.0f), m_scene->GetActiveCam()->GetExtrinsic()));
//     p10 = vec3(CameraToWorld(vec4(p10, 1.0f), m_scene->GetActiveCam()->GetExtrinsic()));
//     p20 = vec3(CameraToWorld(vec4(p20, 1.0f), m_scene->GetActiveCam()->GetExtrinsic()));
//     p30 = vec3(CameraToWorld(vec4(p30, 1.0f), m_scene->GetActiveCam()->GetExtrinsic()));
    
//     WRITE_VEC3(m_renderingZoneVertices, 0, p00);
//     WRITE_VEC3(m_renderingZoneVertices, 3, p10);
//     WRITE_VEC3(m_renderingZoneVertices, 6, p10);
//     WRITE_VEC3(m_renderingZoneVertices, 9, p20);
//     WRITE_VEC3(m_renderingZoneVertices, 12, p20);
//     WRITE_VEC3(m_renderingZoneVertices, 15, p30);
//     WRITE_VEC3(m_renderingZoneVertices, 18, p30);
//     WRITE_VEC3(m_renderingZoneVertices, 21, p00);

//     m_renderZoneLines->UpdateVertices(m_renderingZoneVertices);
// }

const vec3& Volume3D::GetBboxMin(){
    return m_bboxMin;
}

const vec3& Volume3D::GetBboxMax(){
    return m_bboxMax;
}

void Volume3D::Render()
{
    /** nothing special here for now */
    m_lines->Render();
}


std::shared_ptr<CudaLinearVolume3D> Volume3D::GetCudaVolume(){
    return m_cudaVolume;
}