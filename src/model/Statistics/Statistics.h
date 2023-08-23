//
// Created by hepiteau on 27/07/23.
//

#ifndef DRTMCS_STATISTICS_H
#define DRTMCS_STATISTICS_H

#include <queue>
#include "../../view/SceneObject/SceneObject.hpp"
#include "StatsType.h"
#include "ScrollingBuffer.h"

class Statistics : public SceneObject  {
private:
    std::queue<float> m_forwardTime;
    std::queue<float> m_backwardTime;
    std::queue<float> m_batchLoadingTime;

    size_t m_saveAmount = 100;


    ScrollingBuffer m_psnrBuffer;
    double m_t = 0;

    double m_tt[7] = {0,0,0,0,0,0,0};
    ScrollingBuffer m_loadBatchBuffer;
    ScrollingBuffer m_uploadDescBuffer;
    ScrollingBuffer m_zeroGradBuffer;
    ScrollingBuffer m_forwardBuffer;
    ScrollingBuffer m_rayBackwardBuffer;
    ScrollingBuffer m_volBackwardBuffer;
    ScrollingBuffer m_adamUpdateBuffer;

public:

    Statistics();
    Statistics(const Statistics&) = delete;
    ~Statistics() override = default;

    void Set(StatsType type, float value);

    void AddPSNR(float value);
    void AddLoadBatchTime(double value);
    void AddUploadDescTime(double value);
    void AddZeroGradientTime(double value);
    void AddForwardTime(double value);
    void AddRayBackwardTime(double value);
    void AddVolBackwardTime(double value);
    void AddAdamUpdateTime(double value);

    ScrollingBuffer* GetLoadBatchBuffer( );
    ScrollingBuffer* GetUploadDescBuffer( );
    ScrollingBuffer* GetZeroGradientBuffer( );
    ScrollingBuffer* GetForwardBuffer( );
    ScrollingBuffer* GetRayBackwardBuffer( );
    ScrollingBuffer* GetVolBackwardBuffer( );
    ScrollingBuffer* GetAdamUpdateBuffer( );

    double GetTime();
    double* GetTTime();

    ScrollingBuffer* GetPSNRBuffer();

    void Render() override;
};


#endif //DRTMCS_STATISTICS_H
