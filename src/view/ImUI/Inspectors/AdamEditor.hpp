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

        static float arr[] = { 0.6f, 0.1f, 1.0f, 0.5f, 0.92f, 0.1f, 0.2f };
        ImGui::PlotLines("PSNR", arr, IM_ARRAYSIZE(arr), 0, nullptr, 0.0f, 45.0f,  ImVec2(0, 80.0f));

        ImGui::Separator();
    }

};


#endif //ADAM_EDITOR_HPP
