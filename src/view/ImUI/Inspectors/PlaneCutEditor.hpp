/*
Author: Hector Piteau (hector.piteau@gmail.com)
PlaneCutEditor.hpp (c) 2023
Desc: PlaneCutEditor
Created:  2023-04-26T12:46:54.869Z
Modified: 2023-04-26T14:06:53.135Z
*/
#pragma once

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"
#include "../../../include/icons/IconsFontAwesome6.h"
#include "../../../interactors/PlaneCutInteractor.hpp"

#include <glm/glm.hpp>
#include <string>

class PlaneCutEditor {
private:
    PlaneCutInteractor *m_interactor;
public:
    explicit PlaneCutEditor(PlaneCutInteractor *interactor) : m_interactor(interactor) {

    }

    ~PlaneCutEditor() = default;

    void Render() {
        auto dir = m_interactor->GetDirection();
        auto pos = m_interactor->GetPosition();

        ImGui::SeparatorText(ICON_FA_INFO " PlaneCut - Information");
        ImGui::Spacing();

        ImGui::PushID(0);
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4) ImColor::HSV(0.571f, dir == 0 ? 0.6f : 0.0f, 0.6f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4) ImColor::HSV(0.571f, dir == 0 ? 0.7f : 0.0f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4) ImColor::HSV(0.571f, dir == 0 ? 0.8f : 0.0f, 0.8f));
        if (ImGui::Button("   X   ")) {
            m_interactor->SetDirection(PlaneCutDirection::X);
        }
        ImGui::PopStyleColor(3);
        ImGui::PopID();

        ImGui::SameLine();

        ImGui::PushID(1);
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4) ImColor::HSV(0.571f, dir == 1 ? 0.6f : 0.0f, 0.6f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4) ImColor::HSV(0.571f, dir == 1 ? 0.7f : 0.0f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4) ImColor::HSV(0.571f, dir == 1 ? 0.8f : 0.0f, 0.8f));
        if (ImGui::Button("   Y   ")) {
            m_interactor->SetDirection(PlaneCutDirection::Y);
        }
        ImGui::PopStyleColor(3);
        ImGui::PopID();

        ImGui::SameLine();

        ImGui::PushID(2);
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4) ImColor::HSV(0.571f, dir == 2 ? 0.06f : 0.0f, 0.6f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4) ImColor::HSV(0.571f, dir == 2 ? 0.7f : 0.0f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4) ImColor::HSV(0.571f, dir == 2 ? 0.8f : 0.0f, 0.8f));
        if (ImGui::Button("   Z   ")) {
            m_interactor->SetDirection(PlaneCutDirection::Z);
        }
        ImGui::PopStyleColor(3);
        ImGui::PopID();

        if (ImGui::DragFloat("PlaneCut position", &pos, 0.01f)) {
            m_interactor->SetPosition(pos);
        }
    }
};