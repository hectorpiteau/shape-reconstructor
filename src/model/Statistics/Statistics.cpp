//
// Created by hepiteau on 27/07/23.
//

#include "Statistics.h"

Statistics::Statistics() : m_forwardTime(), m_backwardTime(), m_batchLoadingTime() {

}

void Statistics::Set(StatsType type, float value) {
    switch (type) {
        case StatsType::FORWARD:
            m_forwardTime.push(value);
            if(m_forwardTime.size() >= m_saveAmount) m_forwardTime.pop();
            break;
        case StatsType::BACKWARD:
            m_backwardTime.push(value);
            if(m_backwardTime.size() >= m_saveAmount) m_backwardTime.pop();
            break;
        case StatsType::BATCH_LOADING:
            m_batchLoadingTime.push(value);
            if(m_batchLoadingTime.size() >= m_saveAmount) m_batchLoadingTime.pop();
            break;
    }
}
