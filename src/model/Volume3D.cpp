#include <memory>
#include <iostream>
#include <glm/glm.hpp>
#include "../../include/icons/IconsFontAwesome6.h"
#include "Volume3D.hpp"
#include "../cuda/Volume.cuh"

using namespace glm;

Volume3D::Volume3D(Scene *scene, ivec3 res) : SceneObject{std::string("VOLUME3D"), SceneObjectTypes::VOLUME3D},
                                              m_scene(scene), m_res(res), m_desc(), m_volumeDescriptor() {
    SetName(std::string(ICON_FA_CUBES " Volume 3D"));

    m_lines = std::make_shared<Lines>(scene, m_wireframeVertices, 12 * 2 * 3);
    m_lines->SetActive(true);
    ComputeBBoxPoints();
    m_cudaVolume = std::make_shared<CudaLinearVolume3D>(res);
    m_cudaVolume->InitStub();
    m_cudaVolume->ToGPU();

    SetBBoxMin(m_bboxMin);
    SetBBoxMax(m_bboxMax);

    UpdateGPUData();
}

void Volume3D::Resize(const ivec3& res){

}

void Volume3D::DoubleResolution(){
    ivec3 previous_res = m_res;
    m_res *= 2;

    auto new_volume = std::make_shared<CudaLinearVolume3D>(m_res);
    new_volume->InitZeros();
    new_volume->ToGPU();

    volume_resize_double_wrapper(
            m_cudaVolume->GetDevicePtr(),
            new_volume->GetDevicePtr(),
            previous_res,
            m_res);

    m_cudaVolume = new_volume;
    UpdateGPUData();
}

void Volume3D::UpdateGPUData() {
    m_volumeDescriptor.Host()->bboxMin = m_bboxMin;
    m_volumeDescriptor.Host()->bboxMax = m_bboxMax;
    m_volumeDescriptor.Host()->worldSize = m_bboxMax - m_bboxMin;
    m_volumeDescriptor.Host()->res = m_res;
    m_volumeDescriptor.Host()->data = m_cudaVolume->GetDevicePtr();
    m_volumeDescriptor.ToDevice();

    m_desc.Host()->min = m_bboxMin;
    m_desc.Host()->max = m_bboxMax;
    m_desc.ToDevice();
}

void Volume3D::SetBBoxMin(const vec3 &bboxMin) {
    m_bboxMin = bboxMin;
    ComputeBBoxPoints();
    m_lines->UpdateVertices(m_wireframeVertices);

    m_desc.Host()->min = bboxMin;
    m_desc.ToDevice();
}

void Volume3D::SetBBoxMax(const vec3 &bboxMax) {
    m_bboxMax = bboxMax;
    ComputeBBoxPoints();
    m_lines->UpdateVertices(m_wireframeVertices);
    m_desc.Host()->max = bboxMax;
    m_desc.ToDevice();
}

void Volume3D::InitializeZeros() {
    m_cudaVolume->InitZeros();
    m_cudaVolume->ToGPU();
    UpdateGPUData();
}

const ivec3 &Volume3D::GetResolution() {
    return m_res;
}

void Volume3D::ComputeBBoxPoints() {
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

const vec3 &Volume3D::GetBboxMin() {
    return m_bboxMin;
}

const vec3 &Volume3D::GetBboxMax() {
    return m_bboxMax;
}

void Volume3D::Render() {
    /** nothing special here for now */
    m_lines->Render();
}

std::shared_ptr<CudaLinearVolume3D> Volume3D::GetCudaVolume() {
    return m_cudaVolume;
}

BBoxDescriptor *Volume3D::GetGPUDescriptor() {
    return m_desc.Device();
}

GPUData<VolumeDescriptor> &Volume3D::GetGPUData() {
    return m_volumeDescriptor;
}
