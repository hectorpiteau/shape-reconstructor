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

        ImGui::SeparatorText( "Batch");
        ImGui::Spacing();
        if(ImGui::DragInt("Batch-Size", &batchSize, 1)){
            m_interactor->SetBatchSize(batchSize);
        }


        ImGui::Separator();
    }

};


#endif //ADAM_EDITOR_HPP
