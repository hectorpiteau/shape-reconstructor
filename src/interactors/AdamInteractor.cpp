//
// Created by hpiteau on 08/06/23.
//

#include "AdamInteractor.hpp"

void AdamInteractor::SetAdamOptimizer(std::shared_ptr<AdamOptimizer> adam) {
    m_adam = std::move(adam);
}

void AdamInteractor::SetBeta(const vec2& value) {
    return m_adam->SetBeta(value);
}

const vec2& AdamInteractor::GetBeta() const{
    return m_adam->GetBeta();
}

void AdamInteractor::SetEpsilon(float value) {
    m_adam->SetEpsilon(value);
}

float AdamInteractor::GetEpsilon() const {
    return m_adam->GetEpsilon();
}

void AdamInteractor::SetEta(float value) {
    m_adam->SetEta(value);
}

float AdamInteractor::GetEta() const {
    return m_adam->GetEta();
}

unsigned int AdamInteractor::GetBatchSize() {
    return m_adam->GetBatchSize();
}

bool AdamInteractor::IsReady(){
    return m_adam->GetDataLoader()->IsReady();
}

void AdamInteractor::SetBatchSize(unsigned int size){
    m_adam->SetBatchSize(size);
}

bool AdamInteractor::IsOnGPU() {
    return m_adam->GetDataLoader()->IsOnGPU();
}

bool AdamInteractor::IntegrationRangeLoaded(){
    return m_adam->IntegrationRangeLoaded();
}

void AdamInteractor::Initialize(){
    m_adam->Initialize();
}

void AdamInteractor::Optimize(){
    m_adam->Optimize();
}

void AdamInteractor::Step() {
    m_adam->Step();
}

RenderMode AdamInteractor::GetRenderMode(){
    return m_adam->GetRenderMode();
}

void AdamInteractor::SetRenderMode(RenderMode mode) {
    m_adam->SetRenderMode(mode);
}

