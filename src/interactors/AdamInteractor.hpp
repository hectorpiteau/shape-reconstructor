//
// Created by hpiteau on 08/06/23.
//

#ifndef DRTMCS_ADAM_INTERACTOR_HPP
#define DRTMCS_ADAM_INTERACTOR_HPP

#include <memory>
#include "../model/AdamOptimizer.hpp"


class AdamInteractor {
private:
    std::shared_ptr<AdamOptimizer> m_adam;
public:
    AdamInteractor() = default;
    AdamInteractor(const AdamInteractor&) = delete;
    ~AdamInteractor() = default;

    void SetAdamOptimizer(std::shared_ptr<AdamOptimizer> adam);

    void SetBeta(const vec2& value);
    [[nodiscard]] const vec2& GetBeta() const;
    void SetEpsilon(float value);
    [[nodiscard]] float GetEpsilon() const;
    void SetEta(float value);
    [[nodiscard]] float GetEta() const;

    unsigned int GetBatchSize();
    void SetBatchSize(unsigned int size);

    bool IsReady();

    bool IsOnGPU();

    bool IntegrationRangeLoaded();

    void Initialize();

    void Optimize();

    void Step();
};


#endif //DRTMCS_ADAM_INTERACTOR_HPP
