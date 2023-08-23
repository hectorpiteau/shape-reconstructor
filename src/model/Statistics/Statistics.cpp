//
// Created by hepiteau on 27/07/23.
//

#include "Statistics.h"

Statistics::Statistics() : m_forwardTime(), m_backwardTime(), m_batchLoadingTime(), m_psnrBuffer() {
    m_t += ImGui::GetIO().DeltaTime;
    m_psnrBuffer.AddPoint(m_t, 0);
    for(int i=0; i<7; i++) m_tt[i] += ImGui::GetIO().DeltaTime;
     m_loadBatchBuffer.AddPoint(m_tt[0], 0);;
     m_uploadDescBuffer.AddPoint(m_tt[1], 0);;
     m_zeroGradBuffer.AddPoint(m_tt[2], 0);;
     m_forwardBuffer.AddPoint(m_tt[3], 0);;
     m_rayBackwardBuffer.AddPoint(m_tt[4], 0);;
     m_volBackwardBuffer.AddPoint(m_tt[5], 0);;
     m_adamUpdateBuffer.AddPoint(m_tt[6], 0);;
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

void Statistics::AddPSNR(float value) {
    m_t += ImGui::GetIO().DeltaTime;
    m_psnrBuffer.AddPoint(m_t, value);
}

ScrollingBuffer* Statistics::GetPSNRBuffer(){
    return &m_psnrBuffer;
}

void Statistics::Render() {

}

double Statistics::GetTime() {
    return m_t;
}

void Statistics::AddLoadBatchTime(double value) {
    m_tt[0] += ImGui::GetIO().DeltaTime;
    m_loadBatchBuffer.AddPoint(m_tt[0], value);
}

void Statistics::AddUploadDescTime(double value) {
    m_tt[1] += ImGui::GetIO().DeltaTime;
    m_uploadDescBuffer.AddPoint(m_tt[1], value);
}

void Statistics::AddZeroGradientTime(double value) {
    m_tt[2] += ImGui::GetIO().DeltaTime;
    m_zeroGradBuffer.AddPoint(m_tt[2], value);
}

void Statistics::AddForwardTime(double value) {
    m_tt[3] += ImGui::GetIO().DeltaTime;
    m_forwardBuffer.AddPoint(m_tt[3], value);
}

void Statistics::AddRayBackwardTime(double value) {
    m_tt[4] += ImGui::GetIO().DeltaTime;
    m_rayBackwardBuffer.AddPoint(m_tt[4], value);
}

void Statistics::AddVolBackwardTime(double value) {
    m_tt[5] += ImGui::GetIO().DeltaTime;
    m_volBackwardBuffer.AddPoint(m_tt[5], value);
}

void Statistics::AddAdamUpdateTime(double value) {
    m_tt[6] += ImGui::GetIO().DeltaTime;
    m_adamUpdateBuffer.AddPoint(m_tt[6], value);
}

ScrollingBuffer *Statistics::GetLoadBatchBuffer( ) {
    return &m_loadBatchBuffer;
}

ScrollingBuffer *Statistics::GetUploadDescBuffer( ) {
    return &m_uploadDescBuffer;
}

ScrollingBuffer *Statistics::GetZeroGradientBuffer( ) {
    return &m_zeroGradBuffer;
}

ScrollingBuffer *Statistics::GetForwardBuffer() {
    return &m_forwardBuffer;
}

ScrollingBuffer *Statistics::GetRayBackwardBuffer( ) {
    return &m_rayBackwardBuffer;
}

ScrollingBuffer *Statistics::GetVolBackwardBuffer( ) {
    return &m_volBackwardBuffer;
}

ScrollingBuffer *Statistics::GetAdamUpdateBuffer( ) {
    return &m_adamUpdateBuffer;
}

double* Statistics::GetTTime(){
    return m_tt;
}
