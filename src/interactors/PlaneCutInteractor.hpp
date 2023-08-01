#pragma once
#include <memory>
#include <glm/glm.hpp>

#include "../model/PlaneCut.hpp"

using namespace glm;

class PlaneCutInteractor
{
private:
    Scene* m_scene;
    std::shared_ptr<PlaneCut> m_planeCut = nullptr;
    std::vector<std::shared_ptr<Volume3D>> m_availableVolumes;

public:
    explicit PlaneCutInteractor(Scene* scene);
    PlaneCutInteractor(const PlaneCutInteractor &) = delete;

    ~PlaneCutInteractor() = default;

    /**
     * @brief Set the Active PlaneCut to use and edit.
     *
     * @param planeCut : A shared ptr to a PlaneCut instance.
     */
    void SetActivePlaneCut(std::shared_ptr<PlaneCut> planeCut);

    PlaneCutDirection GetDirection();

    void SetDirection(PlaneCutDirection dir);

    void SetPosition(float value);

    float GetPosition();

    vec4 GetCursorValue();

    PlaneCutMode GetMode();

    void SetMode(PlaneCutMode mode);

    std::vector<std::shared_ptr<Volume3D>>& GetAvailableVolumes();

    void SetTargetVolume(std::shared_ptr<Volume3D> vol);
};