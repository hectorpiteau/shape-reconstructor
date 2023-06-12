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