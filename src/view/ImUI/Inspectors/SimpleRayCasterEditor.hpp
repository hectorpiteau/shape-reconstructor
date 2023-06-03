/*
Author: Hector Piteau (hector.piteau@gmail.com)
SimpleRayCasterEditor.hpp (c) 2023
Desc: SimpleRayCasterEditor
Created:  2023-04-26T12:46:54.869Z
Modified: 2023-04-26T14:06:53.135Z
*/
#pragma once
#include <glm/glm.hpp>
#include <string>

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"
#include "../../../include/icons/IconsFontAwesome6.h"

#include "../../../interactors/SimpleRayCasterInteractor.hpp"

class SimpleRayCasterEditor
{
private:
    SimpleRayCasterInteractor* m_interactor;
public:
    SimpleRayCasterEditor(SimpleRayCasterInteractor* interactor) : m_interactor(interactor) {
        
    }
    
    ~SimpleRayCasterEditor() {
        
    }

    void Render(){
        ImGui::SeparatorText(ICON_FA_INFO " Simple RayCaster - Informations");
        bool showRayLines = m_interactor->ShowRayLines();
        int amountOfRays = m_interactor->GetAmountOfRays();
        int rWidth = m_interactor->GetRenderZoneWidth();
        int rHeight = m_interactor->GetRenderZoneHeight();
        
        ImGui::BeginDisabled();
        ImGui::Text("Render zone width (pixels) :");
        ImGui::SameLine();
        ImGui::TextUnformatted(std::to_string(rWidth).c_str());

        ImGui::Text("Render zone height (pixels) :");
        ImGui::SameLine();
        ImGui::TextUnformatted(std::to_string(rHeight).c_str());

        ImGui::InputInt("Amount of rays", &amountOfRays);
        ImGui::EndDisabled();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Checkbox("Show Rays", &showRayLines);

        if(showRayLines != m_interactor->ShowRayLines()){
            m_interactor->SetShowRayLines(showRayLines);
        }
    }
};