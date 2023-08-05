//
// Created by hpiteau on 07/06/23.
//

#include <memory>
#include <utility>
#include "AdamOptimizer.hpp"
#include "Volume/DenseVolume3D.hpp"
#include "Adam.cuh"
#include "Dataset/NeRFDataset.hpp"
#include "Volume.cuh"

using namespace glm;

AdamOptimizer::AdamOptimizer(Scene *scene, std::shared_ptr<Dataset> dataset, std::shared_ptr<DenseVolume3D> target,
                             std::shared_ptr<VolumeRenderer> renderer, std::shared_ptr<SparseVolume3D> sparseVolume) :
        SceneObject{std::string("ADAMOPTIMIZER"), SceneObjectTypes::ADAMOPTIMIZER}, m_scene(scene), m_target(target), m_gradsDescriptor(), m_dataset(std::move(dataset)), m_integrationRangeDescriptor(), m_volumeRenderer(renderer), m_superResModule(4), m_uniformDistribution(0, 8) {
    SetName("Adam Optimizer");
    m_s_target = sparseVolume;

    /** Create the overlay plane that will be used to display the volume rendering texture on. */
    m_overlay = std::make_shared<OverlayPlane>(
            std::make_shared<ShaderPipeline>("../src/shaders/v_overlay_plane.glsl",
                                             "../src/shaders/f_overlay_plane.glsl"), scene->GetSceneSettings());

    /** Create the cuda texture that will receive the result of the volume rendering process. */
    m_cudaTex = std::make_shared<CudaTexture>(
            scene->GetSceneSettings()->GetViewportWidth(),
            scene->GetSceneSettings()->GetViewportHeight());

    /** ********** ADAM G1 ********** */
    m_adamG1 = std::make_shared<DenseVolume3D>(scene, target->GetResolution());
    m_adamG1->SetName("Adam G1");
    m_adamG1->InitializeZeros();
    m_scene->Add(m_adamG1, true, true);
    m_children.push_back(m_adamG1);

    m_s_adamG1 = std::make_shared<SparseVolume3D>(sparseVolume->GetInitialResolution());
    m_s_adamG1->SetName("Sparse Adam G1");
    m_s_adamG1->InitializeZeros();
    m_scene->Add(m_s_adamG1, true, true);
    m_children.push_back(m_s_adamG1);
    /** ********** ADAM G1 ********** */


    /** ********** ADAM G2 ********** */
    m_adamG2 = std::make_shared<DenseVolume3D>(scene, target->GetResolution());
    m_adamG2->SetName("Adam G2");
    m_adamG2->InitializeZeros();
    m_scene->Add(m_adamG2, true, true);
    m_children.push_back(m_adamG2);

    m_s_adamG2 = std::make_shared<SparseVolume3D>(sparseVolume->GetInitialResolution());
    m_s_adamG2->SetName("Sparse Adam G2");
    m_s_adamG2->InitializeZeros();
    m_scene->Add(m_s_adamG2, true, true);
    m_children.push_back(m_s_adamG2);
    /** ********** ADAM G2 ********** */


    /** ********** ADAM GRADIENTS ********** */
    m_grads = std::make_shared<DenseVolume3D>(scene, target->GetResolution());
    m_grads->SetName("Gradients");
    m_scene->Add(m_grads, true, true);
    m_children.push_back(m_grads);

    m_s_grads = std::make_shared<SparseVolume3D>(sparseVolume->GetInitialResolution());
    m_s_grads->SetName("Sparse Adam Gradients");
    m_s_grads->InitializeZeros();
    m_scene->Add(m_s_grads, true, true);
    m_children.push_back(m_s_grads);
    /** ********** ADAM GRADIENTS ********** */

    m_dataLoader = std::make_shared<DataLoader>(m_dataset);

    m_integrationRangeDescriptor.Host()->surface = m_cudaTex->GetTex();
    m_integrationRangeDescriptor.ToDevice();
}

//void AdamOptimizer::SetTargetDataVolume(std::shared_ptr<DenseVolume3D> targetVolume){
//    m_target = std::move(targetVolume);
//}

void AdamOptimizer::Initialize() {
    UpdateGPUDescriptor();

    /** Initialize integration ranges. */
    auto cameraSet = m_dataset->GetCameraSet();
    for (const auto &cam: cameraSet->GetCameras()) {
        cam->UpdateGPUDescriptor();
        cam->GetIntegrationRangeGPUDescriptor().Host()->data = (float2 *) GPUData<IntegrationRangeDescriptor>::AllocateOnDevice(
                cam->GetResolution().x * cam->GetResolution().y * sizeof(float2));
        cam->GetIntegrationRangeGPUDescriptor().Host()->dim.x = cam->GetResolution().x;
        cam->GetIntegrationRangeGPUDescriptor().Host()->dim.y = cam->GetResolution().y;
        cam->GetIntegrationRangeGPUDescriptor().Host()->renderInTexture = false;

        cam->GetIntegrationRangeGPUDescriptor().Host()->surface = cam->GetCudaTexture()->OpenSurface();
        cam->GetIntegrationRangeGPUDescriptor().ToDevice();

        integration_range_bbox_wrapper(cam->GetGPUData(), cam->GetIntegrationRangeGPUDescriptor().Device(),
                                       m_target->GetBBoxGPUDescriptor());
//        cam->GetCudaTexture()->RunKernel(m_volumeRenderer->GetRayCasterGPUData(), cam->GetGPUData(),
//                                         m_volumeRenderer->GetVolumeGPUData());
        cam->GetCudaTexture()->CloseSurface();
    }

    m_integrationRangeLoaded = true;

    /** Initialize dataset. */
    m_dataLoader->Initialize(m_superResModule.GetRaysAmount());

//    zero_adam_wrapper(&m_adamDescriptor);
    sparse_zero_adam_wrapper(&m_s_adamDescriptor);
}

bool AdamOptimizer::IntegrationRangeLoaded() const {
    return m_integrationRangeLoaded;
}

void AdamOptimizer::Optimize() {
    if (!m_integrationRangeLoaded) {
        std::cerr << "Adam Optimizer:  Cannot optimize, integration ranges are not computed." << std::endl;
        return;
    }

    m_optimize = !m_optimize;
}

void AdamOptimizer::Render() {
    if (m_optimize) {
        Step();
    }
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

void AdamOptimizer::Step() {
    m_dataLoader->LoadBatch(m_renderMode);

    /** Update adam descriptor's data on GPU. */
    UpdateGPUDescriptor();
    m_superResModule.Step();
    m_volumeRenderer->UpdateGPUDescriptors();

    /** Zero gradients. */
//    zero_adam_wrapper(&m_adamDescriptor);
    sparse_zero_adam_wrapper(&m_s_adamDescriptor);

    /** Forward Pass.  */
    auto items = m_dataLoader->GetGPUDatas();
    for (size_t i = 0; i < m_dataLoader->GetBatchSize(); ++i) {
//        batched_forward_wrapper(*items[i], m_volumeRenderer->GetVolumeGPUData(), m_superResModule.GetDescriptor());
        batched_forward_sparse_wrapper(*items[i], m_s_target->GetDescriptor());
    }

    /** Backward Pass.  */
    for (size_t i = 0; i < m_dataLoader->GetBatchSize(); ++i) {
//        batched_backward_wrapper(*items[i], m_volumeRenderer->GetVolumeGPUData(), m_adamDescriptor, m_superResModule.GetDescriptor());
        batched_backward_sparse_wrapper(
                items[i],
                m_s_target->GetDescriptor(),
                &m_s_adamDescriptor,
                m_superResModule.GetDescriptor());
    }

    /** Volume Backward. Computing gradients on voxels and not on image rays. */
//    volume_backward((GPUData<DenseVolumeDescriptor>*)(m_volumeRenderer->GetVolumeGPUData()), &m_adamDescriptor);
    sparse_volume_backward((GPUData<SparseVolumeDescriptor>*)(m_s_target->GetDescriptor()), &m_s_adamDescriptor);

//    items[0]->ToHost();
//    auto mse = items[0]->Host()->psnr.x / (float)(items[0]->Host()->res.x * items[0]->Host()->res.y);
//    auto psnr = 10.0f * log10(1.0f / mse);
//    std::cout << "PSNR x: " << std::to_string(items[0]->Host()->psnr.x) << " / " << std::to_string(psnr) << " / " << std::to_string(items[0]->Host()->res.x) << std::endl;

    /** Update target volume weights. */
//    update_adam_wrapper(&m_adamDescriptor);
    sparse_update_adam_wrapper(&m_s_adamDescriptor);

    m_steps += 1;

    m_dataLoader->UnloadBatch();
    m_dataLoader->NextBatch();
}

void AdamOptimizer::NextLOD() {
    /** If already the maximum level of detail, nothing happens. */
    if (m_currentLODIndex == LOD_AMOUNT - 1) return;
    m_currentLODIndex += 1;

    /** Augment volume's resolutions. */
    m_target->DoubleResolution();
    m_adamG1->DoubleResolution();
    m_adamG2->DoubleResolution();
    m_grads->DoubleResolution();

    /** Augment images resolutions. */
    m_dataset = std::make_shared<NeRFDataset>(m_scene,
                                              LODs[m_currentLODIndex].json_train_path,
                                              LODs[m_currentLODIndex].image_train_path,
                                              LODs[m_currentLODIndex].json_valid_path,
                                              LODs[m_currentLODIndex].image_valid_path);
    m_dataset->Load();

    m_dataLoader = std::make_shared<DataLoader>(m_dataset);
    m_dataLoader->Initialize(m_superResModule.GetRaysAmount());
}

void AdamOptimizer::UpdateGPUDescriptor() {
    m_gradsDescriptor.Host()->data = m_grads->GetCudaVolume()->GetDevicePtr();
    m_gradsDescriptor.Host()->bboxMin = m_grads->GetBboxMin();
    m_gradsDescriptor.Host()->bboxMax = m_grads->GetBboxMax();
    m_gradsDescriptor.Host()->worldSize = m_grads->GetBboxMax() - m_grads->GetBboxMin();
    m_gradsDescriptor.Host()->res = m_grads->GetResolution();
    m_gradsDescriptor.ToDevice();

//    m_adamDescriptor.Host()->epsilon = m_epsilon;
//    m_adamDescriptor.Host()->eta = m_eta;
//    m_adamDescriptor.Host()->adamG1 = m_adamG1->GetCudaVolume()->GetDevicePtr();
//    m_adamDescriptor.Host()->adamG2 = m_adamG2->GetCudaVolume()->GetDevicePtr();
//    m_adamDescriptor.Host()->target = m_target->GetCudaVolume()->GetDevicePtr();
//    m_adamDescriptor.Host()->grads = m_gradsDescriptor.Device();
//    m_adamDescriptor.Host()->iteration = (int) m_steps;
//    m_adamDescriptor.Host()->res = m_target->GetResolution();
//    m_adamDescriptor.Host()->color_0_w = GetColor0W();
//    m_adamDescriptor.Host()->alpha_0_w = GetAlpha0W();
//    m_adamDescriptor.Host()->alpha_reg_0_w = GetAlphaReg0W();
//    m_adamDescriptor.Host()->tvl2_0_w = GetTVL20W();
//    m_adamDescriptor.Host()->amountOfGradientsToWrite = m_amountOfGradientsToWrite;

    m_s_adamDescriptor.Host()->epsilon = m_epsilon;
    m_s_adamDescriptor.Host()->eta = m_eta;
    m_s_adamDescriptor.Host()->beta = m_beta;
    m_s_adamDescriptor.Host()->res = m_s_target->GetResolution();
    m_s_adamDescriptor.Host()->grads = m_s_grads->GetDescriptor()->Device();
    m_s_adamDescriptor.Host()->adamG1 = m_s_adamG1->GetDescriptor()->Device();
    m_s_adamDescriptor.Host()->adamG2 = m_s_adamG2->GetDescriptor()->Device();
    m_s_adamDescriptor.Host()->target = m_s_target->GetDescriptor()->Device();
    m_s_adamDescriptor.Host()->iteration = m_steps;
    m_s_adamDescriptor.Host()->color_0_w = GetColor0W();
    m_s_adamDescriptor.Host()->alpha_0_w = GetAlpha0W();
    m_s_adamDescriptor.Host()->alpha_reg_0_w = GetAlphaReg0W();
    m_s_adamDescriptor.Host()->tvl2_0_w = GetTVL20W();
    m_s_adamDescriptor.Host()->amountOfGradientsToWrite = m_amountOfGradientsToWrite;

    for(int i=0; i < m_amountOfGradientsToWrite; i++) {
//        m_adamDescriptor.Host()->writeGradientIndexes[i] = m_uniformDistribution.Get();
        m_s_adamDescriptor.Host()->writeGradientIndexes[i] = m_uniformDistribution.Get();
    }

//    m_adamDescriptor.ToDevice();
    m_s_adamDescriptor.ToDevice();
}

void AdamOptimizer::SetBeta(const vec2 &value) {
    m_beta = value;
}

const vec2 &AdamOptimizer::GetBeta() const {
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

std::shared_ptr<DenseVolume3D> AdamOptimizer::GetGradVolume() {
    return m_grads;
}

void AdamOptimizer::SetRenderMode(RenderMode mode) {
    m_renderMode = mode;
}

RenderMode AdamOptimizer::GetRenderMode() {
    return m_renderMode;
}

std::shared_ptr<DenseVolume3D> AdamOptimizer::GetTargetVolume() {
    return m_target;
}

void AdamOptimizer::SetUseSuperResolution(bool value) {
    m_superResModule.SetActive(value);
}

bool AdamOptimizer::UseSuperResolution() {
    return m_superResModule.IsActive();
}

SuperResolutionModule* AdamOptimizer::GetSuperResolutionModule(){
    return &m_superResModule;
}

void AdamOptimizer::CullVolume(){
    sparse_volume_cull_wrapper(m_s_target->GetDescriptor());
}

void AdamOptimizer::DivideVolume(){
    m_s_target->GetDescriptor()->Host()->occupiedVoxelCount = 0;
    m_s_target->GetDescriptor()->ToDevice();
    std::cout << "Voxel count before : " << std::to_string(m_s_target->GetDescriptor()->Host()->occupiedVoxelCount) << std::endl;
    sparse_volume_divide_wrapper(m_s_target->GetDescriptor());
    m_s_target->GetDescriptor()->ToHost();
    std::cout << "Voxel count after : " << std::to_string(m_s_target->GetDescriptor()->Host()->occupiedVoxelCount) << std::endl;
}