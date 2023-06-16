//
// Created by hpiteau on 07/06/23.
//

#include <memory>
#include <glm/glm.hpp>
#include <utility>
#include "AdamOptimizer.hpp"
#include "../cuda/CudaLinearVolume3D.cuh"
#include "Volume3D.hpp"

using namespace glm;

AdamOptimizer::AdamOptimizer(Scene* scene, std::shared_ptr<Dataset> dataset, std::shared_ptr<VolumeRenderer> volumeRenderer, const ivec3 &volumeResolution) :
        SceneObject{std::string("ADAMOPTIMIZER"), SceneObjectTypes::ADAMOPTIMIZER}, m_scene(scene), m_res(volumeResolution),m_dataset(std::move(dataset)), m_integrationRangeDescriptor(), m_volumeRenderer(std::move(volumeRenderer)) {
    SetName("Adam Optimizer");

    m_target = m_volumeRenderer->GetVolume3D();

    /** Create the overlay plane that will be used to display the volume rendering texture on. */
    m_overlay = std::make_shared<OverlayPlane>(
            std::make_shared<ShaderPipeline>("../src/shaders/v_overlay_plane.glsl",
                                             "../src/shaders/f_overlay_plane.glsl"), scene->GetSceneSettings());

    /** Create the cuda texture that will receive the result of the volume rendering process. */
    m_cudaTex = std::make_shared<CudaTexture>(
            scene->GetSceneSettings()->GetViewportWidth(),
            scene->GetSceneSettings()->GetViewportHeight());

    m_adamG1 = std::make_shared<CudaLinearVolume3D>(volumeResolution);
    m_adamG2 = std::make_shared<CudaLinearVolume3D>(volumeResolution);
    m_blurredVoxels = std::make_shared<CudaLinearVolume3D>(volumeResolution);
    m_dataLoader = std::make_shared<DataLoader>(m_dataset->GetCameraSet(), m_dataset->GetImageSet());

    m_integrationRangeDescriptor.Host()->surface = m_cudaTex->GetTex();
    m_integrationRangeDescriptor.ToDevice();
}

void AdamOptimizer::SetTargetDataVolume(std::shared_ptr<Volume3D> targetVolume){
    m_target = std::move(targetVolume);
}

void AdamOptimizer::Initialize(){


    /** Initialize integration ranges. */
    auto cameraSet = m_dataset->GetCameraSet();
    for(const auto& cam : cameraSet->GetCameras()){

        cam->GetIntegrationRangeGPUDescriptor().Host()->data = (float2*)GPUData<IntegrationRangeDescriptor>::AllocateOnDevice(cam->GetResolution().x * cam->GetResolution().y * sizeof(float2));
        cam->GetIntegrationRangeGPUDescriptor().Host()->dim.x = cam->GetResolution().x;
        cam->GetIntegrationRangeGPUDescriptor().Host()->dim.y = cam->GetResolution().y;
        cam->GetIntegrationRangeGPUDescriptor().Host()->renderInTexture = false;

//        cam->GetIntegrationRangeGPUDescriptor().Host()->surface = cam->GetCudaTexture()->OpenSurface();
        cam->GetIntegrationRangeGPUDescriptor().ToDevice();

        integration_range_bbox_wrapper(cam->GetGPUData(), cam->GetIntegrationRangeGPUDescriptor().Device(), m_target->GetGPUDescriptor());
//        cam->UpdateGPUDescriptor();
//        cam->GetCudaTexture()->RunKernel(m_volumeRenderer->GetRayCasterGPUData(), cam->GetGPUData(), m_volumeRenderer->GetVolumeGPUData());

//        cam->GetCudaTexture()->CloseSurface();
    }

    m_integrationRangeLoaded = true;

    /** Initialize dataset. */
    m_dataLoader->Initialize();
    m_dataLoader->LoadNext();


}

bool AdamOptimizer::IntegrationRangeLoaded() const{
    return m_integrationRangeLoaded;
}

void AdamOptimizer::Optimize(){
    Step();
}

void AdamOptimizer::Render() {
    //nothing for now.
    if(m_optimize){
        Step();
    }
//


//    m_integrationRangeDescriptor.Host()->dim = ivec2(m_scene->GetSceneSettings()->GetViewportWidth(),
//                                                     m_scene->GetSceneSettings()->GetViewportHeight());
//    m_integrationRangeDescriptor.Host()->renderInTexture = true;
//
//    m_scene->GetActiveCam()->UpdateGPUDescriptor();
//
//    m_cudaTex->RunCUDAIntegralRange(m_integrationRangeDescriptor, m_scene->GetActiveCam()->GetGPUData(), m_target->GetGPUDescriptor());
//    m_overlay->Render(true, m_cudaTex->GetTex(), m_scene);

//    m_testPlane->SetUseCustomTex(true);
//    m_testPlane->SetCustomTex(m_cudaTex->GetTex());
//    m_testPlane->Render();
}

void AdamOptimizer::Step(){
    /**  */

}

void AdamOptimizer::UpdateGPUDescriptor() {
    m_adamDescriptor.Host()->epsilon = m_epsilon;
    m_adamDescriptor.Host()->eta = m_eta;
    m_adamDescriptor.Host()->adamG1 = m_adamG1->GetDevicePtr();
    m_adamDescriptor.Host()->adamG2 = m_adamG1->GetDevicePtr();
    m_adamDescriptor.Host()->target = m_target->GetCudaVolume()->GetDevicePtr();
}

void AdamOptimizer::SetBeta(const vec2& value) {
    m_beta = value;
}

const vec2& AdamOptimizer::GetBeta() const {
    return m_beta;
}

void AdamOptimizer::SetEpsilon(float value) {
    m_epsilon = value;
}

float AdamOptimizer::GetEpsilon() const {
    return m_epsilon;
}

void AdamOptimizer::SetEta(float value) {
    m_eta = value;
}

float AdamOptimizer::GetEta() const {
    return m_eta;
}

std::shared_ptr<DataLoader> AdamOptimizer::GetDataLoader() {
    return m_dataLoader;
}

void AdamOptimizer::SetBatchSize(unsigned int batchSize) {
    return m_dataLoader->SetBatchSize(batchSize);
}

unsigned int AdamOptimizer::GetBatchSize() {
    return m_dataLoader->GetBatchSize();
}
