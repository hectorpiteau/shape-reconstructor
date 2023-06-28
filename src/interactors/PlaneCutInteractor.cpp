#include <memory>
#include <utility>
#include "PlaneCutInteractor.hpp"
#include "../model/PlaneCut.hpp"

void PlaneCutInteractor::SetActivePlaneCut(std::shared_ptr<PlaneCut> planeCut)
{
    m_planeCut = std::move(planeCut);
}

//std::shared_ptr<PlaneCut> PlaneCutInteractor::GetActivePlaneCut()
//{
//    return m_planeCut;
//}

PlaneCutDirection PlaneCutInteractor::GetDirection()
{
    return m_planeCut->GetDirection();
}

void PlaneCutInteractor::SetDirection(PlaneCutDirection dir)
{
    m_planeCut->SetDirection(dir);
}

void PlaneCutInteractor::SetPosition(float value)
{
    m_planeCut->SetPosition(value);
}

float PlaneCutInteractor::GetPosition()
{
    return m_planeCut->GetPosition();
}

vec4 PlaneCutInteractor::GetCursorValue() {
    return m_planeCut->GetCursorPixelValue();
}

PlaneCutMode PlaneCutInteractor::GetMode(){
    return m_planeCut->GetMode();
}

void PlaneCutInteractor::SetMode(PlaneCutMode mode){
    m_planeCut->SetMode(mode);
}

PlaneCutInteractor::PlaneCutInteractor(Scene *scene) : m_scene(scene), m_availableVolumes() {
    auto res = m_scene->GetAll(SceneObjectTypes::VOLUME3D);

    for(const auto& volume : res)
        m_availableVolumes.push_back(std::dynamic_pointer_cast<Volume3D>(volume));
}

std::vector<std::shared_ptr<Volume3D>> &PlaneCutInteractor::GetAvailableVolumes() {
    return m_availableVolumes;
}

void PlaneCutInteractor::SetTargetVolume(std::shared_ptr<Volume3D> vol) {
    m_planeCut->SetTargetVolume(vol);
}
