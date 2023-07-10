//
// Created by hpiteau on 08/06/23.
//
#ifndef ADAM_EDITOR_HPP
#define ADAM_EDITOR_HPP

#include <memory>

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"
#include "../../../include/icons/IconsFontAwesome6.h"

#include "../../../interactors/AdamInteractor.hpp"


#define RADIO_BUTTON(RDMODE, BTN_NAME) if(mode == RDMODE) ImGui::BeginDisabled();\
if(ImGui::Button(BTN_NAME)){\
m_interactor->SetRenderMode(RDMODE);\
}\
if(mode == RDMODE) ImGui::EndDisabled();

class AdamEditor {
private:
    AdamInteractor *m_interactor;
public:
    explicit AdamEditor(AdamInteractor *interactor) {
        m_interactor = interactor;
    }
    AdamEditor(const AdamEditor &) = delete;
    ~AdamEditor() = default;

    void Render() {
        if (m_interactor == nullptr) {
            ImGui::Text("Error: interactor is null.");
            return;
        }

        auto beta = m_interactor->GetBeta();
        auto eps = m_interactor->GetEpsilon();
        auto eta = m_interactor->GetEta();
        auto batchSize = m_interactor->GetBatchSize();
        auto isReady = m_interactor->IsReady();
        auto isOnGPU = m_interactor->IsOnGPU();
        auto intRangeLoaded = m_interactor->IntegrationRangeLoaded();
        auto mode = m_interactor->GetRenderMode();
        auto color0w = m_interactor->GetColor0W();
        auto alpha0w = m_interactor->GetAlpha0W();
        auto alphareg0w = m_interactor->GetAlphaReg0W();
        auto tvl20w = m_interactor->GetTVL20W();

        ImGui::SeparatorText(ICON_FA_INFO " Adam Optimizer - Information");
        ImGui::Spacing();

        if(ImGui::DragFloat2("Beta", &beta[0], 0.001f)){
            m_interactor->SetBeta(beta);
        }

        if(ImGui::DragFloat("Epsilon", &eps, 0.001f)){
            m_interactor->SetEpsilon(eps);
        }

        if(ImGui::DragFloat("Eta", &eta, 0.001f)){
            m_interactor->SetEta(eta);
        }

        ImGui::SeparatorText( "DataLoader");
        ImGui::Spacing();

        ImGui::Text("IsReady: ");
        ImGui::SameLine();
        if(isReady){
            ImGui::Text("Yes - Camera&Image Set Ok");
        }else{
            ImGui::Text("No - Camera&Image Set Not Ok");
        }

        ImGui::Text("IsBatchOnGPU: ");
        ImGui::SameLine();
        if(isOnGPU){
            ImGui::Text("Yes");
        }else{
            ImGui::Text("No");
        }

        if(ImGui::DragInt("Batch-Size", (int*)&batchSize, 1, 1)){
            m_interactor->SetBatchSize(batchSize);
        }

        /** RENDER MODE */

        RADIO_BUTTON(RenderMode::GROUND_TRUTH, "Ground Truth")
        ImGui::SameLine();
        RADIO_BUTTON(RenderMode::PREDICTED_COLOR, "Color Pred")
        ImGui::SameLine();
        RADIO_BUTTON(RenderMode::PREDICTED_TRANSMIT, "Transmit Pred")

        RADIO_BUTTON(RenderMode::COLOR_LOSS, "Color Loss")
        ImGui::SameLine();
        RADIO_BUTTON(RenderMode::ALPHA_LOSS, "Transmit Loss")



        ImGui::SeparatorText( "Actions");
        ImGui::Spacing();

        if(ImGui::Button("Initialize", ImVec2(ImGui::GetWindowSize().x*0.96f, 30.0f))){
            m_interactor->Initialize();
        }
        ImGui::Text("Integration Ranges: ");
        ImGui::SameLine();
        if(intRangeLoaded){
            ImGui::Text("Loaded");
        }else{
            ImGui::Text("Not Loaded");
        }
        ImGui::Spacing();
        if(ImGui::Button("Step", ImVec2(ImGui::GetWindowSize().x*0.96f, 30.0f))){
            m_interactor->Step();
        }
        ImGui::Spacing();
        if(ImGui::Button("Optimize", ImVec2(ImGui::GetWindowSize().x*0.96f, 30.0f))){
            m_interactor->Optimize();
        }

        ImGui::SeparatorText( "Stats");
        ImGui::Spacing();
        ImGui::Text("Amount of iterations: ");
        ImGui::SameLine();
        ImGui::TextUnformatted(std::to_string(0).c_str());

        static float arr[] = { 0.0f };
        ImGui::PlotLines("PSNR", arr, IM_ARRAYSIZE(arr), 0, nullptr, 0.0f, 45.0f,  ImVec2(0, 80.0f));

        ImGui::Separator();

        ImGui::SeparatorText( "Loss weighting");
        ImGui::Spacing();

        if(ImGui::DragFloat("Color L2", &color0w, 0.001f)){
            m_interactor->SetColor0W(color0w);
        }

        if(ImGui::DragFloat("Alpha L2", &alpha0w, 0.001f)){
            m_interactor->SetAlpha0W(alpha0w);
        }

        if(ImGui::DragFloat("Alpha Reg", &alphareg0w, 0.001f)){
            m_interactor->SetAlphaReg0W(alphareg0w);
        }

        if(ImGui::DragFloat("TVL2 Reg", &tvl20w, 0.001f)){
            m_interactor->SetTVL20W(tvl20w);
        }

    }

};


#endif //ADAM_EDITOR_HPP
