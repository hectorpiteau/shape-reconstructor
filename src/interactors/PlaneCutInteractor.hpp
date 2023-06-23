#pragma once
#include <memory>
#include <glm/glm.hpp>

#include "../model/PlaneCut.hpp"

using namespace glm;

class PlaneCutInteractor
{
private:
    std::shared_ptr<PlaneCut> m_planeCut = nullptr;

public:
    PlaneCutInteractor() = default;
    PlaneCutInteractor(const PlaneCutInteractor &) = delete;

    ~PlaneCutInteractor() = default;

    /**
     * @brief Set the Active PlaneCut to use and edit.
     *
     * @param planeCut : A shared ptr to a PlaneCut instance.
     */
    void SetActivePlaneCut(std::shared_ptr<PlaneCut> planeCut);

    /**
     * @brief Get the Active PlaneCut instance.
     *
     * @return std::shared_ptr<PlaneCut> : The currently active PlaneCut.
     */
//    std::shared_ptr<PlaneCut> GetActivePlaneCut();

    PlaneCutDirection GetDirection();

    void SetDirection(PlaneCutDirection dir);

    void SetPosition(float value);

    float GetPosition();

    vec4 GetCursorValue();
};