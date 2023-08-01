//
// Created by hepiteau on 27/07/23.
//

#ifndef DRTMCS_STATISTICS_H
#define DRTMCS_STATISTICS_H

#include <queue>
#include "../../view/SceneObject/SceneObject.hpp"
#include "StatsType.h"

class Statistics : public SceneObject  {
private:
    std::queue<float> m_forwardTime;
    std::queue<float> m_backwardTime;
    std::queue<float> m_batchLoadingTime;

    size_t m_saveAmount = 100;

public:

    Statistics();
    Statistics(const Statistics&) = delete;
    ~Statistics() = default;

    void Set(StatsType type, float value);

};


#endif //DRTMCS_STATISTICS_H
