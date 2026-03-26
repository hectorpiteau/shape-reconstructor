//
// Created by hepiteau on 23/08/23.
//

#ifndef RT3DRS_STATSVIEW_H
#define RT3DRS_STATSVIEW_H
#include <memory>
#include "../../model/Statistics/Statistics.h"

class StatsView {
private:
    std::shared_ptr<Statistics>  m_stats;
public:
    explicit StatsView(std::shared_ptr<Statistics> statistics);
    StatsView(const StatsView&) = delete;
    void Render();
};
#endif //RT3DRS_STATSVIEW_H
