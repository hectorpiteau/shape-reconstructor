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

void AdamInteractor::SetColor0W(float value) {
    m_adam->SetColor0W(value);
}

void AdamInteractor::SetAlpha0W(float value) {
    m_adam->SetAlpha0W(value);
}

void AdamInteractor::SetAlphaReg0W(float value) {
    m_adam->SetAlphaReg0W(value);
}

float AdamInteractor::GetColor0W() {
    return m_adam->GetColor0W();
}

float AdamInteractor::GetAlpha0W() {
    return m_adam->GetAlpha0W();
}

float AdamInteractor::GetAlphaReg0W() {
    return m_adam->GetAlphaReg0W();
}

float AdamInteractor::GetTVL20W() {
    return m_adam->GetTVL20W();
}

void AdamInteractor::NextLOD(){
    m_adam->NextLOD();
}

void AdamInteractor::SetTVL20W(float value) {
    m_adam->SetTVL20W(value);
}

void AdamInteractor::SetUseSuperResolution(bool value) {
    m_adam->SetUseSuperResolution(value);
}

bool AdamInteractor::UseSuperResolution() {
    return m_adam->UseSuperResolution();
}

SuperResolutionModule* AdamInteractor::GetSuperResolutionModule() {
    return m_adam->GetSuperResolutionModule();
}

void AdamInteractor::CullVolume() {
    m_adam->CullVolume();
}

void AdamInteractor::DivideVolume() {
    m_adam->DivideVolume();
}
